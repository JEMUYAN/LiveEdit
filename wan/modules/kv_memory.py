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
    """Policy-free adapter over LiveEdit's per-layer cache dictionary."""

    def __init__(self, cache: MutableMapping[str, Any]):
        self.cache = cache

    @property
    def capacity(self) -> int:
        return self.cache["k"].shape[1]

    @property
    def global_end(self) -> int:
        return self.cache["global_end_index"].item()

    @property
    def local_end(self) -> int:
        return self.cache["local_end_index"].item()

    def move_left(
        self,
        *,
        preserved_prefix: int,
        evicted_tokens: int,
        moved_tokens: int,
    ) -> None:
        """Move a cache region left; the caller owns the eviction policy."""

        source_start = preserved_prefix + evicted_tokens
        source_end = source_start + moved_tokens
        destination_end = preserved_prefix + moved_tokens
        self.cache["k"][:, preserved_prefix:destination_end] = (
            self.cache["k"][:, source_start:source_end].clone()
        )
        self.cache["v"][:, preserved_prefix:destination_end] = (
            self.cache["v"][:, source_start:source_end].clone()
        )

    def write(
        self,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        end = start + key.shape[1]
        if start < 0 or end > self.capacity:
            raise RuntimeError(
                f"KV write [{start}:{end}] exceeds cache capacity {self.capacity}"
            )
        self.cache["k"][:, start:end] = key
        self.cache["v"][:, start:end] = value

    def read(self, start: int, end: int) -> SelectedKV:
        return SelectedKV(
            self.cache["k"][:, start:end],
            self.cache["v"][:, start:end],
        )

    def read_ranges(self, ranges: list[tuple[int, int]]) -> SelectedKV:
        selected = [self.read(start, end) for start, end in ranges]
        if len(selected) == 1:
            return selected[0]
        return SelectedKV(
            torch.cat([item.key for item in selected], dim=1),
            torch.cat([item.value for item in selected], dim=1),
        )

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
    ) -> SelectedKV:
        """Select an ordered logical view without mutating the KV store."""


class HistoryPolicy(HistorySelector):
    """Interface that owns cache update and historical-KV selection decisions."""

    @abstractmethod
    def update(
        self,
        store: KVStore,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        current_start: int,
        frame_seqlen: int,
    ) -> KVUpdate:
        """Apply the policy's cache update and return the resulting indices."""


class NoMemoryManagementPolicy(HistoryPolicy):
    """Baseline: append in order and expose all history without eviction."""

    def update(
        self,
        store: KVStore,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        current_start: int,
        frame_seqlen: int,
    ) -> KVUpdate:
        del frame_seqlen
        current_end = current_start + key.shape[1]
        local_end = store.local_end + current_end - store.global_end
        local_start = local_end - key.shape[1]
        store.write(local_start, key, value)
        return KVUpdate(current_end, local_start, local_end)

    def select(
        self,
        store: KVStore,
        *,
        local_end: int,
        frame_seqlen: int,
    ) -> SelectedKV:
        del frame_seqlen
        return store.read(0, local_end)


class SinkRecentHistoryPolicy(HistoryPolicy):
    """Official LiveEdit policy: rolling cache, sink, and recent history."""

    def __init__(
        self,
        local_attn_size: int,
        max_attention_size: int,
        sink_size: int,
    ):
        self.local_attn_size = local_attn_size
        self.max_attention_size = max_attention_size
        self.sink_size = sink_size

    def update(
        self,
        store: KVStore,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        current_start: int,
        frame_seqlen: int,
    ) -> KVUpdate:
        current_end = current_start + key.shape[1]
        num_new_tokens = key.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        should_roll = (
            self.local_attn_size != -1
            and current_end > store.global_end
            and num_new_tokens + store.local_end > store.capacity
        )
        if should_roll:
            num_evicted_tokens = (
                num_new_tokens + store.local_end - store.capacity
            )
            num_rolled_tokens = (
                store.local_end - num_evicted_tokens - sink_tokens
            )
            store.move_left(
                preserved_prefix=sink_tokens,
                evicted_tokens=num_evicted_tokens,
                moved_tokens=num_rolled_tokens,
            )
            local_end = (
                store.local_end
                + current_end
                - store.global_end
                - num_evicted_tokens
            )
        else:
            local_end = store.local_end + current_end - store.global_end

        local_start = local_end - num_new_tokens
        store.write(local_start, key, value)
        return KVUpdate(current_end, local_start, local_end)

    def select(
        self,
        store: KVStore,
        *,
        local_end: int,
        frame_seqlen: int,
    ) -> SelectedKV:
        if self.local_attn_size == -1:
            attention_start = max(0, local_end - self.max_attention_size)
            return store.read(attention_start, local_end)

        logical_cache_tokens = self.local_attn_size * frame_seqlen
        sink_tokens = self.sink_size * frame_seqlen
        recent_tokens = logical_cache_tokens - sink_tokens
        if local_end <= logical_cache_tokens:
            return store.read(0, local_end)
        if sink_tokens == 0:
            return store.read(local_end - recent_tokens, local_end)
        return store.read_ranges([
            (0, sink_tokens),
            (local_end - recent_tokens, local_end),
        ])


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
