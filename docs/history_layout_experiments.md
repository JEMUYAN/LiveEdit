# LiveEdit historical-layout experiments

## Question and isolation boundary

The experiment asks how much freedom the released LiveEdit checkpoint tolerates
in the *identity and order* of historical KV chunks. It does not train or alter
the backbone, checkpoint, QKV projections, or model parameters.

For controlled comparisons, `ChunkLayoutHistoryPolicy` uses an append-only full
KV cache and changes only the ordered ranges returned to attention. This is
important: if an experiment simultaneously evicted a chunk and changed the
selected layout, a quality change could not be attributed to selection alone.
The official `SinkRecentHistoryPolicy` remains the default and retains its
original rolling behavior.

The experimental path has three layers:

1. `KVStore` retains K/V and exposes tensor ranges.
2. `HistoryLayoutPlanner` returns ordered chunk ids using pure Python.
3. `PositionMapper` applies the model's existing absolute or window-relative
   RoPE semantics to the selected KV.

The current chunk is always included. Reordering therefore changes historical
KV order but keeps current Q at the final logical position under window-relative
RoPE. Each run writes `history_trace.json`; use it to verify the *realized*
layout rather than relying only on the requested configuration.

## Experimental factors

Use the official Sink+Recent run with the same case and seed as the primary
baseline. Keep checkpoint, prompt, source-frame sampling, denoising steps,
noise seed, output length, resolution, and RoPE mode fixed.

| Factor | Planner/config | Controlled comparison |
|---|---|---|
| Full-history interface baseline | `no_memory_management` | Confirms the policy-free interface and measures the effect of truncation. |
| Non-contiguous history | `stride` or `explicit` | Vary stride and number of retained chunks at fixed recent context. |
| Cross-scene history | `scene_contrast` | Combine recent chunks with the most different annotated scenes. |
| Number of initial sinks | official policy + `sink_size` sweep | Sweep sink frames while holding total `local_attn_size` fixed. |
| Non-initial/variable sinks | `anchor_recent` | Replace chunk zero with fixed or per-current-chunk anchors. |
| Historical order | planner `order` | Compare chronological, explicit selection order, and reverse-history order. |

`sink_size` in the official implementation is measured in latent frames;
planner anchor IDs are causal append units. With the released local config,
`num_frame_per_block=3`, so a 21-latent-frame sample normally has seven chunk
IDs. Always confirm this in the trace because a different config (for example
an independent first frame) changes the mapping.

### Recommended staged matrix

Do not start with the full Cartesian product. Run these stages so failures are
diagnosable and compute is bounded:

1. Smoke: one case, one seed, official baseline plus full history, stride-2,
   sink counts 0/3/6, and one non-initial anchor.
2. Seed stability: the same case with at least three seeds.
3. Content coverage: local edits, global edits, low/high motion, single/multiple
   scene clips, and edits whose target appears late.
4. Dense boundary sweep near the first visible degradation, with at least five
   seeds and multiple cases.

For a claim of statistical difference, use at least 20 paired case-seed samples
where practical. A three-seed smoke test is useful for debugging but not a
reliable significance claim.

## Data manifest

Start from `experiments/history_layout/example_manifest.json`. A case supports:

```json
{
  "id": "stable-id",
  "source_path": "/data/source.mp4",
  "instruction": "Change the white coat to black.",
  "reference_path": "/data/optional-edited-reference.mp4",
  "mask_path": "/data/optional-edit-region-mask.mp4",
  "scene_by_chunk": {"0": "shot-a", "1": "shot-a", "2": "shot-b"}
}
```

`mask_path` uses white for the intended edit region; preservation metrics are
then computed outside that region. Without a mask, source similarity is a
full-frame diagnostic and can penalize a successful global edit. Scene labels
must describe causal chunk IDs, not raw video-frame indices.

Existing LiveEdit JSON can be imported without copying videos:

```bash
python experiments/history_layout/import_liveedit_dataset.py \
  test_cases/long_test.json /tmp/liveedit_cases.json
```

Copy the emitted case objects into a manifest, then add scene labels, masks, or
references as available.

For graded rather than binary scene difference, add `scene_distance` to the
`scene_contrast` planner, for example `{"indoor|beach": 0.9}`. The planner
selects the largest annotated distances, breaks ties by recency, and writes the
realized IDs to the trace. Without this map every different label has distance
one. Alternatively use an `explicit` schedule for fully controlled pairs.

## Server workflow

The runner deliberately launches one inference process per case/variant/seed.
This is slower to initialize but isolates CUDA failures, makes resumption safe,
and matches the current 24GB-memory cleanup path, which moves the generator to
CPU before VAE decoding.

First validate and inspect every planned command without importing PyTorch:

```bash
python experiments/history_layout/run_experiments.py \
  experiments/history_layout/example_manifest.json --dry-run
```

On the GPU server:

```bash
python experiments/history_layout/run_experiments.py \
  /path/to/manifest.json --resume
```

Each run directory contains the singleton inference input, exact policy JSON,
command, Git commit, log, generated video, and realized history trace. Failed
runs retain their log and nonzero return code.

## Evaluation

The public repository does not contain the standalone scoring scripts used for
the paper's quantitative tables. The framework therefore provides transparent
metrics using dependencies already declared by LiveEdit, and leaves room for
an external benchmark adapter rather than presenting them as the authors'
unreleased evaluator.

Default metrics:

- source PSNR/similarity/MAE, either full-frame or outside an edit mask;
- source-motion delta error, measuring how much frame-to-frame changes depart
  from the source motion;
- the same pixel metrics against an optional edited reference; and
- wall-clock time and failure status recorded by the runner.

Optional `--clip` adds frame-text CLIP cosine similarity using `open_clip_torch`.
It is disabled by default because pretrained weights may not already be cached.
For publication-grade results, add VBench/EditBoard or a human preference study
as a separate evaluator while preserving the same per-run JSON contract.

Run and summarize:

```bash
python experiments/history_layout/evaluate.py /path/to/output-root --clip
python experiments/history_layout/summarize.py /path/to/output-root \
  --baseline official_sink3_recent6 \
  --minimum-effect clip_instruction_alignment=0.01
```

Outputs are `per_sample.csv`, `aggregate.csv`, `paired_deltas.csv`,
`summary.json`, and `report.md`. Paired deltas match case, seed, and metric.
They are sign-normalized so positive always favors the candidate. The report
uses a deterministic 95% paired bootstrap interval; an interval excluding zero
is marked directional. Report effect size and confidence interval together,
and predeclare a practically meaningful threshold before calling a difference
important.

## Interpretation guardrails

- Absolute RoPE preserves each selected chunk's original time coordinate;
  window-relative RoPE reassigns coordinates according to the packed selection
  order. Treat these as separate experiments.
- Scene-contrast results require manual scene annotation or a separately
  versioned detector. Do not infer semantic distance from chunk number.
- Full-cache policies increase VRAM versus the official rolling cache. Record
  peak VRAM separately; quality comparisons isolate selection, not efficiency.
- Keep internal token pruning disabled for this experiment set. Arbitrary
  chunk-range storage currently requires complete latent-frame KV, and the
  code raises explicitly if a pruned token layout is supplied.
- A layout may improve CLIP alignment while damaging preservation or temporal
  stability. Use the metric vector and Pareto frontier, not a single average.
- Inspect videos and traces for the first failures. Metric changes alone cannot
  distinguish intended stronger editing from background leakage.

## Files added or changed

- `wan/modules/history_layout.py`: dependency-free planners.
- `wan/modules/kv_memory.py`: append-only chunk-layout policy and traces.
- `wan/modules/history_experiment.py`: JSON policy binding.
- `pipeline/causal_inference.py`: optional cache-capacity override and trace
  capture before server-side memory cleanup.
- `inference-mm.py`: opt-in policy and trace CLI arguments.
- `experiments/history_layout/`: import, orchestration, evaluation, aggregation,
  and an example manifest.
- `tests/test_history_layout_planners.py` and
  `tests/test_history_experiment_manifest.py`: laptop-safe tests.
