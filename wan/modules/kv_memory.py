"""Parameter-free KV storage, history selection, and position mapping.

These adapters preserve LiveEdit's existing cache dictionary and tensor
operations. They separate policy from attention without changing the
checkpoint-visible module tree.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional

import torch


@dataclass(frozen=True)
class KVUpdate:
    """Indices produced by one physical cache update."""

    current_end: int
    local_start: int
    local_end: int


@dataclass(frozen=True)
class SelectedKV:
    """Ordered logical KV view passed to attention."""

    key: torch.Tensor
    value: torch.Tensor


class KVStore:
    """Thin adapter over LiveEdit's existing per-layer cache dictionary."""

    def __init__(self, cache: MutableMapping[str, Any]):
        self.cache = cache

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        current_start: int,
        local_attn_size: int,
        sink_tokens: int,
    ) -> KVUpdate:
        current_end = current_start + key.shape[1]
        global_end = self.cache["global_end_index"].item()
        local_end = self.cache["local_end_index"].item()
        cache_size = self.cache["k"].shape[1]
        num_new_tokens = key.shape[1]

        should_roll = (
            local_attn_size != -1
            and current_end > global_end
            and num_new_tokens + local_end > cache_size
        )
        if should_roll:
            num_evicted_tokens = num_new_tokens + local_end - cache_size
            num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
            self.cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                self.cache["k"][:,
                                sink_tokens + num_evicted_tokens:
                                sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            )
            self.cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = (
                self.cache["v"][:,
                                sink_tokens + num_evicted_tokens:
                                sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            )
            next_local_end = (
                local_end + current_end - global_end - num_evicted_tokens
            )
        else:
            next_local_end = local_end + current_end - global_end

        local_start = next_local_end - num_new_tokens
        self.cache["k"][:, local_start:next_local_end] = key
        self.cache["v"][:, local_start:next_local_end] = value
        return KVUpdate(current_end, local_start, next_local_end)

    def commit(self, update: KVUpdate) -> None:
        """Commit indices after the attention operation has succeeded."""

        self.cache["global_end_index"].fill_(update.current_end)
        self.cache["local_end_index"].fill_(update.local_end)


class HistorySelector(ABC):
    """Interface for deciding which ordered KV view attention receives."""

    @abstractmethod
    def select(
        self,
        store: KVStore,
        *,
        local_end: int,
        frame_seqlen: int,
        sink_size: int,
    ) -> SelectedKV:
        """Select an ordered logical view without mutating the KV store."""


class SinkRecentHistorySelector(HistorySelector):
    """Official LiveEdit policy: persistent sink plus newest recent frames."""

    def __init__(self, local_attn_size: int, max_attention_size: int):
        self.local_attn_size = local_attn_size
        self.max_attention_size = max_attention_size

    def select(
        self,
        store: KVStore,
        *,
        local_end: int,
        frame_seqlen: int,
        sink_size: int,
    ) -> SelectedKV:
        cache = store.cache
        if self.local_attn_size == -1:
            attention_start = max(0, local_end - self.max_attention_size)
            return SelectedKV(
                cache["k"][:, attention_start:local_end],
                cache["v"][:, attention_start:local_end],
            )

        logical_cache_tokens = self.local_attn_size * frame_seqlen
        sink_tokens = sink_size * frame_seqlen
        recent_tokens = logical_cache_tokens - sink_tokens
        if local_end <= logical_cache_tokens:
            return SelectedKV(
                cache["k"][:, :local_end],
                cache["v"][:, :local_end],
            )
        if sink_tokens == 0:
            return SelectedKV(
                cache["k"][:, local_end - recent_tokens:local_end],
                cache["v"][:, local_end - recent_tokens:local_end],
            )
        return SelectedKV(
            torch.cat([
                cache["k"][:, :sink_tokens],
                cache["k"][:, local_end - recent_tokens:local_end],
            ], dim=1),
            torch.cat([
                cache["v"][:, :sink_tokens],
                cache["v"][:, local_end - recent_tokens:local_end],
            ], dim=1),
        )


class PositionMapper:
    """Map global or logical-window frame positions to RoPE coordinates."""

    def __init__(self, window_rope: bool):
        self.window_rope = window_rope

    def prepare_current(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        current_start_frame: int,
        is_pruned: bool,
        pruning_info: Optional[Mapping[str, Any]],
        rope_apply: Callable[..., torch.Tensor],
        rope_apply_pruned: Callable[..., torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.window_rope and is_pruned:
            raise NotImplementedError(
                "window_rope is not compatible with pruned KV tokens yet"
            )
        if self.window_rope:
            return None, key
        if is_pruned:
            roped_query = rope_apply_pruned(
                query, grid_sizes, freqs, pruning_info,
                start_frame=current_start_frame,
            ).type_as(value)
            cache_key = rope_apply_pruned(
                key, grid_sizes, freqs, pruning_info,
                start_frame=current_start_frame,
            ).type_as(value)
        else:
            roped_query = rope_apply(
                query, grid_sizes, freqs,
                start_frame=current_start_frame,
            ).type_as(value)
            cache_key = rope_apply(
                key, grid_sizes, freqs,
                start_frame=current_start_frame,
            ).type_as(value)
        return roped_query, cache_key

    def map_selected(
        self,
        query: torch.Tensor,
        selected_key: torch.Tensor,
        value: torch.Tensor,
        roped_query: Optional[torch.Tensor],
        *,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        frame_seqlen: int,
        rope_apply: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.window_rope:
            return roped_query, selected_key
        if selected_key.shape[1] % frame_seqlen != 0:
            raise ValueError(
                "window_rope requires a whole number of frames in KV cache, "
                f"got {selected_key.shape[1]} tokens for frame_seqlen={frame_seqlen}"
            )

        key_grid_sizes = grid_sizes.clone()
        key_grid_sizes[:, 0] = selected_key.shape[1] // frame_seqlen
        mapped_key = rope_apply(
            selected_key, key_grid_sizes, freqs, start_frame=0,
        ).type_as(value)
        query_window_start = (
            mapped_key.shape[1] - query.shape[1]
        ) // frame_seqlen
        mapped_query = rope_apply(
            query, grid_sizes, freqs, start_frame=query_window_start,
        ).type_as(value)
        return mapped_query, mapped_key
