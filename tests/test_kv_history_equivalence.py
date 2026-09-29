"""Server-side equivalence tests for the KV history refactor.

`legacy_step` is a compact, frozen copy of the pre-refactor cached-attention
branch. It intentionally does not call the new classes.
"""

import importlib.util
import math
import unittest
from pathlib import Path

import torch


def _load_kv_memory():
    """Load the policy module without importing the ``wan`` package.

    ``python tests/test_kv_history_equivalence.py`` puts ``tests/`` on
    ``sys.path``, not the repository root. Importing ``wan`` also executes
    ``T5EncoderModel``'s default ``torch.cuda.current_device()``, which fails
    when no GPU is visible. This test only needs the parameter-free KV module.
    """

    module_path = (
        Path(__file__).resolve().parents[1] / "wan" / "modules" / "kv_memory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "kv_memory_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_kv_memory = _load_kv_memory()
KVStore = _kv_memory.KVStore
ChunkLayoutHistoryPolicy = _kv_memory.ChunkLayoutHistoryPolicy
NoMemoryManagementPolicy = _kv_memory.NoMemoryManagementPolicy
PositionMapper = _kv_memory.PositionMapper
SinkRecentHistoryPolicy = _kv_memory.SinkRecentHistoryPolicy
build_history_layout_planner = _kv_memory.build_history_layout_planner


def make_cache(capacity, heads=2, head_dim=4):
    return {
        "k": torch.zeros(1, capacity, heads, head_dim),
        "v": torch.zeros(1, capacity, heads, head_dim),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
    }


def clone_cache(cache):
    return {name: value.clone() for name, value in cache.items()}


def fake_rope(x, grid_sizes, freqs, start_frame=0):
    """Small deterministic position transform with the same call contract."""

    del freqs
    frames, height, width = grid_sizes[0].tolist()
    positions = torch.arange(
        start_frame,
        start_frame + frames,
        dtype=x.dtype,
        device=x.device,
    ).repeat_interleave(height * width)
    return x + positions.view(1, -1, 1, 1)


def unused_pruned_rope(*args, **kwargs):
    raise AssertionError("pruned RoPE should not be called in this test")


def scaled_dot_attention(query, key, value):
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
    scores = scores / math.sqrt(query.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", weights, value)


def legacy_step(
    cache,
    query,
    key,
    value,
    *,
    current_start,
    frame_seqlen,
    local_attn_size,
    sink_size,
    max_attention_size,
    window_rope,
):
    grid_sizes = torch.tensor([[query.shape[1] // frame_seqlen, 1, frame_seqlen]])
    freqs = torch.empty(0)
    current_start_frame = current_start // frame_seqlen

    if not window_rope:
        roped_query = fake_rope(
            query, grid_sizes, freqs, start_frame=current_start_frame
        ).type_as(value)
        cache_key = fake_rope(
            key, grid_sizes, freqs, start_frame=current_start_frame
        ).type_as(value)
    else:
        roped_query = None
        cache_key = key

    current_end = current_start + query.shape[1]
    sink_tokens = sink_size * frame_seqlen
    cache_size = cache["k"].shape[1]
    num_new_tokens = query.shape[1]
    if (
        local_attn_size != -1
        and current_end > cache["global_end_index"].item()
        and num_new_tokens + cache["local_end_index"].item() > cache_size
    ):
        num_evicted_tokens = (
            num_new_tokens + cache["local_end_index"].item() - cache_size
        )
        num_rolled_tokens = (
            cache["local_end_index"].item()
            - num_evicted_tokens
            - sink_tokens
        )
        cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache["k"][:,
            sink_tokens + num_evicted_tokens:
            sink_tokens + num_evicted_tokens + num_rolled_tokens
        ].clone()
        cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache["v"][:,
            sink_tokens + num_evicted_tokens:
            sink_tokens + num_evicted_tokens + num_rolled_tokens
        ].clone()
        local_end = (
            cache["local_end_index"].item()
            + current_end
            - cache["global_end_index"].item()
            - num_evicted_tokens
        )
    else:
        local_end = (
            cache["local_end_index"].item()
            + current_end
            - cache["global_end_index"].item()
        )
    local_start = local_end - num_new_tokens
    cache["k"][:, local_start:local_end] = cache_key
    cache["v"][:, local_start:local_end] = value

    if local_attn_size == -1:
        attention_start = max(0, local_end - max_attention_size)
        attention_k = cache["k"][:, attention_start:local_end]
        attention_v = cache["v"][:, attention_start:local_end]
    else:
        logical_cache_tokens = local_attn_size * frame_seqlen
        recent_tokens = logical_cache_tokens - sink_tokens
        if local_end <= logical_cache_tokens:
            attention_k = cache["k"][:, :local_end]
            attention_v = cache["v"][:, :local_end]
        elif sink_tokens == 0:
            attention_k = cache["k"][:, local_end - recent_tokens:local_end]
            attention_v = cache["v"][:, local_end - recent_tokens:local_end]
        else:
            attention_k = torch.cat([
                cache["k"][:, :sink_tokens],
                cache["k"][:, local_end - recent_tokens:local_end],
            ], dim=1)
            attention_v = torch.cat([
                cache["v"][:, :sink_tokens],
                cache["v"][:, local_end - recent_tokens:local_end],
            ], dim=1)

    if window_rope:
        key_grid_sizes = grid_sizes.clone()
        key_grid_sizes[:, 0] = attention_k.shape[1] // frame_seqlen
        attention_k = fake_rope(
            attention_k, key_grid_sizes, freqs, start_frame=0
        ).type_as(value)
        query_window_start = (
            attention_k.shape[1] - query.shape[1]
        ) // frame_seqlen
        roped_query = fake_rope(
            query, grid_sizes, freqs, start_frame=query_window_start
        ).type_as(value)

    output = scaled_dot_attention(roped_query, attention_k, attention_v)
    cache["global_end_index"].fill_(current_end)
    cache["local_end_index"].fill_(local_end)
    return output, attention_k, attention_v


def refactored_step(
    cache,
    query,
    key,
    value,
    *,
    current_start,
    frame_seqlen,
    local_attn_size,
    sink_size,
    max_attention_size,
    window_rope,
):
    grid_sizes = torch.tensor([[query.shape[1] // frame_seqlen, 1, frame_seqlen]])
    freqs = torch.empty(0)
    mapper = PositionMapper(window_rope)
    roped_query, cache_key = mapper.prepare_current(
        query,
        key,
        value,
        grid_sizes=grid_sizes,
        freqs=freqs,
        current_start_frame=current_start // frame_seqlen,
        is_pruned=False,
        pruning_info=None,
        rope_apply=fake_rope,
        rope_apply_pruned=unused_pruned_rope,
    )
    store = KVStore(cache)
    policy = SinkRecentHistoryPolicy(
        local_attn_size,
        max_attention_size,
        sink_size,
    )
    update = policy.update(
        store,
        cache_key,
        value,
        current_start=current_start,
        frame_seqlen=frame_seqlen,
    )
    selected = policy.select(
        store,
        local_end=update.local_end,
        frame_seqlen=frame_seqlen,
    )
    roped_query, attention_k = mapper.map_selected(
        query,
        selected.key,
        value,
        roped_query,
        grid_sizes=grid_sizes,
        freqs=freqs,
        frame_seqlen=frame_seqlen,
        rope_apply=fake_rope,
    )
    output = scaled_dot_attention(roped_query, attention_k, selected.value)
    store.commit(update)
    return output, attention_k, selected.value


class KVHistoryEquivalenceTest(unittest.TestCase):
    def test_original_and_refactored_cached_attention_are_equivalent(self):
        cases = [
            (10, 5, 1, 32760, False, [0, 4, 8, 12]),
            (10, 5, 0, 32760, False, [0, 4, 8, 12]),
            (16, 5, 1, 32760, False, [0, 4, 8]),
            (16, -1, 0, 8, False, [0, 4, 8]),
            (10, 5, 1, 32760, True, [0, 4, 8, 12]),
            # Re-evaluate a committed chunk, as diffusion inference does.
            (10, 5, 1, 32760, True, [0, 4, 8, 8, 12]),
        ]
        for case in cases:
            with self.subTest(case=case):
                self._run_case(*case)

    def _run_case(
        self,
        capacity,
        local_attn_size,
        sink_size,
        max_attention_size,
        window_rope,
        starts,
    ):
        torch.manual_seed(20260928)
        legacy_cache = make_cache(capacity)
        refactored_cache = clone_cache(legacy_cache)

        for current_start in starts:
            query = torch.randn(1, 4, 2, 4)
            key = torch.randn(1, 4, 2, 4)
            value = torch.randn(1, 4, 2, 4)
            legacy = legacy_step(
                legacy_cache,
                query,
                key,
                value,
                current_start=current_start,
                frame_seqlen=2,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                max_attention_size=max_attention_size,
                window_rope=window_rope,
            )
            refactored = refactored_step(
                refactored_cache,
                query,
                key,
                value,
                current_start=current_start,
                frame_seqlen=2,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                max_attention_size=max_attention_size,
                window_rope=window_rope,
            )

            for legacy_tensor, refactored_tensor in zip(legacy, refactored):
                torch.testing.assert_close(
                    legacy_tensor, refactored_tensor, rtol=0, atol=0
                )
            for name in legacy_cache:
                torch.testing.assert_close(
                    legacy_cache[name], refactored_cache[name], rtol=0, atol=0
                )

    def test_no_memory_management_baseline_appends_and_exposes_all_history(self):
        cache = make_cache(capacity=8)
        store = KVStore(cache)
        policy = NoMemoryManagementPolicy()

        first_key = torch.full((1, 4, 2, 4), 1.0)
        first_value = torch.full((1, 4, 2, 4), 10.0)
        first = policy.update(
            store,
            first_key,
            first_value,
            current_start=0,
            frame_seqlen=2,
        )
        store.commit(first)

        second_key = torch.full((1, 4, 2, 4), 2.0)
        second_value = torch.full((1, 4, 2, 4), 20.0)
        second = policy.update(
            store,
            second_key,
            second_value,
            current_start=4,
            frame_seqlen=2,
        )
        selected = policy.select(
            store,
            local_end=second.local_end,
            frame_seqlen=2,
        )
        store.commit(second)

        torch.testing.assert_close(
            selected.key,
            torch.cat([first_key, second_key], dim=1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            selected.value,
            torch.cat([first_value, second_value], dim=1),
            rtol=0,
            atol=0,
        )

        with self.assertRaisesRegex(RuntimeError, "exceeds cache capacity"):
            policy.update(
                store,
                torch.zeros(1, 2, 2, 4),
                torch.zeros(1, 2, 2, 4),
                current_start=8,
                frame_seqlen=2,
            )

    def test_chunk_layout_selects_non_contiguous_ranges_and_traces_once(self):
        cache = make_cache(capacity=12)
        store = KVStore(cache)
        planner = build_history_layout_planner({
            "type": "explicit",
            "schedule": {"2": [0, 2]},
        })
        policy = ChunkLayoutHistoryPolicy(planner)
        values = []
        for chunk_id, current_start in enumerate((0, 4, 8)):
            key = torch.full((1, 4, 2, 4), float(chunk_id + 1))
            value = torch.full((1, 4, 2, 4), float((chunk_id + 1) * 10))
            update = policy.update(
                store, key, value,
                current_start=current_start, frame_seqlen=2,
            )
            selected = policy.select(
                store, local_end=update.local_end, frame_seqlen=2,
            )
            store.commit(update)
            values.append((key, value, selected))

        torch.testing.assert_close(
            values[-1][2].key,
            torch.cat([values[0][0], values[2][0]], dim=1),
            rtol=0, atol=0,
        )
        self.assertEqual(
            cache["history_selection_trace"][-1]["selected_chunks"], [0, 2]
        )

        # A repeated diffusion step overwrites current KV but creates neither a
        # duplicate chunk record nor a duplicate realized-layout trace.
        repeated = policy.update(
            store, values[-1][0], values[-1][1],
            current_start=8, frame_seqlen=2,
        )
        policy.select(store, local_end=repeated.local_end, frame_seqlen=2)
        self.assertEqual(len(cache["history_chunks"]), 3)
        self.assertEqual(len(cache["history_selection_trace"]), 3)


if __name__ == "__main__":
    unittest.main()
