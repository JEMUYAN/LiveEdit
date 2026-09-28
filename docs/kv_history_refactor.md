# KV history selection refactor

## Goal

Extract the decision "which historical KV should this query attend to?" from
`CausalWanSelfAttention.forward()` without changing:

- model parameters or checkpoint keys;
- cache allocation and the externally visible cache dictionary;
- training behavior when `kv_cache is None`;
- the default inference path;
- sink + recent rolling-window semantics; or
- absolute and window-relative RoPE semantics.

This first phase deliberately adds no learned policy, training objective, or
new runtime configuration.

## Baseline implementation map

At baseline commit `77a9fcc6ee2e2d84d232877e4494f9e8c563514a`:

- `pipeline/causal_inference.py::_initialize_kv_cache()` and the analogous
  pipeline methods allocate one dictionary per transformer block. Each
  dictionary contains `k`, `v`, `global_end_index`, and `local_end_index`.
- `wan/modules/causal_model.py::CausalWanSelfAttention.forward()` projects the
  current chunk to Q/K/V, applies or defers RoPE, rolls the physical cache,
  writes current K/V, selects the logical attention view, remaps window-relative
  positions, runs attention, and commits both cache indices.
- `utils/attention_mask.py::blockwise_causal_attention_mask()` implements the
  corresponding sink + recent visibility rule for non-cached blockwise
  FlexAttention training.

The cached path has two coordinate systems:

- `global_end_index` is the end token index in the complete generated video;
- `local_end_index` is the end token index in the physical rolling cache.

The physical cache remains compatible with existing pipelines. This refactor
only gives that dictionary a small adapter (`KVStore`).

### Baseline configuration caveat

The implementation treats `local_attn_size` as the total logical window:

```text
recent_size = local_attn_size - sink_size
```

`infer-local-ar-forcing-long.sh` currently comments "12-frame logical window =
3 sink + 9 recent", but its defaults pass `local_attn_size=9` and `sink_size=3`.
The executable code therefore produces 3 sink + 6 recent frames. Phase one
preserves the executable behavior rather than silently changing it to match the
comment. Any intended configuration correction should be a separate change.

## Components

### `KVStore`

Provides policy-free physical storage primitives only: read ranges, write a
range, move a range left, expose capacity/current indices, and commit updated
indices after attention succeeds. It does not decide whether to evict, which
prefix to preserve, where new KV belongs, or which tokens are visible.

### `HistorySelector` and `HistoryPolicy`

`HistorySelector` defines the ordered-view interface. `HistoryPolicy` extends
it with the cache-update decision. A policy therefore owns both halves of
memory management: how history is retained and which retained history is used.

### `NoMemoryManagementPolicy`

This is the baseline implementation used to validate the abstraction itself.
It writes each new chunk sequentially and exposes the entire initialized prefix.
It does not roll, evict, preserve a sink, or truncate/select recent history. If
the preallocated cache is too small, it raises an explicit capacity error rather
than silently applying a memory policy.

### `SinkRecentHistoryPolicy`

This is the author's original LiveEdit behavior, moved behind `HistoryPolicy`.
It owns all of the following:

1. detecting a new global append;
2. deciding when the physical cache must roll;
3. preserving the first `S` sink frames;
4. evicting the oldest non-sink tokens;
5. writing the new chunk at the resulting local position; and
6. selecting the ordered logical view used by attention.

For a bounded window of `W` frames and a sink of `S` frames:

```text
visible = first S frames + newest (W - S) frames
```

Before the logical window is full, it returns the contiguous initialized prefix.
For `local_attn_size == -1`, it preserves the existing
`max_attention_size=32760` tail selection.

### `PositionMapper`

Keeps position semantics separate from storage/selection:

- absolute RoPE rotates current Q/K using global frame positions before K is
  stored; selected cached K is already rotated;
- window-relative RoPE stores raw K, selects/rolls it first, then rotates
  selected K from window frame zero and current Q at its location in that window;
- pruned KV remains unsupported with window-relative RoPE, matching baseline.

The mapper owns no tensors, parameters, or persistent state.

`CausalWanSelfAttention` binds `SinkRecentHistoryPolicy` by default, so official
inference behavior stays unchanged. The bound object is a plain Python object,
not an `nn.Module`; it adds no checkpoint keys or learned parameters. A caller
can replace `history_policy` with `NoMemoryManagementPolicy` for a full-history
baseline when the allocated cache is large enough.

## Compatibility invariants

The refactored path must preserve at every chunk:

1. identical `kv_cache["k"]` and `kv_cache["v"]` contents;
2. identical `global_end_index` and `local_end_index`;
3. identical selected attention K/V, including concatenation order;
4. identical RoPE frame coordinates for Q/K;
5. identical attention input shapes, dtypes, and devices; and
6. no additions to `state_dict()`.

The public constructor and `forward()` signatures are unchanged. Existing
checkpoint loading therefore does not require migration.

## Server-side validation

The Mac checkout intentionally does not install PyTorch or project dependencies.
Run the focused test inside the existing LiveEdit server environment:

```bash
git switch refactor/kv-history-selection
python tests/test_kv_history_equivalence.py -v
```

The test uses a fixed seed and compares a frozen copy of the original inline
algorithm with `SinkRecentHistoryPolicy` after every chunk. It covers cache
growth, rolling, sink retention, no-sink selection, unbounded selection,
absolute RoPE, window-relative RoPE, and final scaled-dot-product attention. It
also verifies that `NoMemoryManagementPolicy` appends/exposes all history and
raises on capacity exhaustion instead of evicting. Expected result: exact
equality for cache/index tensors and zero numerical difference for the official
policy path, since both paths execute the same operations in the same order.

Then run one existing short inference command twice from the same checkpoint
and seed: once at the baseline commit and once on this branch. Save the latent
output before VAE decode and compare:

```python
diff = (baseline_latent.float() - refactor_latent.float()).abs()
print("max_abs", diff.max().item())
print("mean_abs", diff.mean().item())
print("equal", torch.equal(baseline_latent, refactor_latent))
```

Acceptance criteria for phase one:

- the focused equivalence test passes;
- checkpoint loading reports no missing or unexpected keys;
- generated latent shapes/dtypes match;
- `max_abs == 0` is expected for a deterministic backend; if the backend is
  nondeterministic, record deterministic settings and compare against the
  baseline's own repeat-run variation before attributing a difference to this
  refactor; and
- peak VRAM and per-chunk cache length do not increase.

## Future extension boundary

A later phase may add alternative `HistorySelector` implementations. They
should consume cache metadata and return an ordered KV view; they must not alter
backbone projections, add trainable parameters, or silently change positional
mapping. That work is intentionally outside this phase.
