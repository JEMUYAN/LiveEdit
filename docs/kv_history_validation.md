# KV history refactor validation record

## Baseline and scope

- Baseline commit: `77a9fcc6ee2e2d84d232877e4494f9e8c563514a`
- Development branch: `refactor/kv-history-selection`
- Runtime environment changed on Mac: no
- Conda environment or project dependency installed on Mac: no
- Checkpoint/model inference run on Mac: no

## Mac-side checks completed

| Check | Result | Notes |
| --- | --- | --- |
| Python AST parse | PASS | Parsed the changed Python files without importing PyTorch. |
| Git whitespace check | PASS | `git diff --check` returned no errors. |
| Public attention signatures | UNCHANGED | Constructor and `forward()` signatures were not modified. |
| Pipeline cache schema | UNCHANGED | Existing `k`, `v`, `global_end_index`, and `local_end_index` dictionaries remain in use. |
| Policy boundary | PASS BY INSPECTION | `KVStore` contains storage primitives; rolling, sink retention, eviction, and logical selection live in `SinkRecentHistoryPolicy`. |
| Baseline behavior | PASS BY INSPECTION | `NoMemoryManagementPolicy` only appends and returns all initialized KV; it raises rather than evicts at capacity. |
| Model parameters | UNCHANGED BY DESIGN | Policy objects are plain Python classes, not `nn.Module` or `nn.Parameter`. Confirm on server by comparing state-dict keys. |

These checks establish source-level consistency only. They are not runtime or
numerical-equivalence evidence.

### Configuration discrepancy retained intentionally

The code defines recent frames as `local_attn_size - sink_size`. The long-video
script defaults to `local_attn_size=9`, `sink_size=3` while its comment describes
3 sink + 9 recent frames. The refactor preserves the code's effective 3 + 6
behavior. Do not mix a configuration correction into the equivalence test.

## Server-side focused equivalence test

Run from the repository root in the existing LiveEdit environment:

```bash
python tests/test_kv_history_equivalence.py -v
```

The test has no dependency on pytest. It uses `torch.manual_seed(20260928)` and
compares the original and refactored algorithms after every chunk with
`rtol=0, atol=0` for:

- physical cache K/V;
- global/local cache indices;
- selected attention K/V;
- absolute and window-relative position mapping; and
- scaled-dot-product attention output.

The same file separately checks the no-management baseline: sequential append,
full-history visibility, and an explicit error on capacity exhaustion.

Record the result here after the server run:

```text
Date:
Server/GPU:
PyTorch/CUDA:
Command:
Result:
Failure details (if any):
```

## Full-model A/B record

Use the same checkpoint, prompt/input, seed, inference parameters, and backend
settings for baseline and refactor. Prefer saving and comparing the final latent
before VAE decoding, because video encoding can introduce unrelated differences.

```text
Checkpoint:
Input:
Seed:
Inference command/config:
Baseline commit:
Refactor commit:
Baseline repeat max_abs (nondeterminism control):
Baseline vs refactor max_abs:
Baseline vs refactor mean_abs:
torch.equal:
Missing/unexpected checkpoint keys:
Peak VRAM baseline/refactor:
Per-chunk KV lengths baseline/refactor:
Conclusion:
```

## Files involved

- `wan/modules/causal_model.py`: delegates the cached-attention branch.
- `wan/modules/kv_memory.py`: policy-free store, policy interfaces, no-management baseline, official Sink+Recent policy, and position mapper.
- `tests/test_kv_history_equivalence.py`: fixed-seed reference comparison.
- `docs/kv_history_refactor.md`: design and invariants.
- `docs/kv_history_validation.md`: validation status and server record.
