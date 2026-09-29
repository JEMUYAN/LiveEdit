#!/usr/bin/env python3
"""Evaluate generated videos with reproducible, per-sample metrics.

Pixel metrics are dependency-light and run with the repository environment.
CLIP is optional because its pretrained weights may need a separate download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_video(path: str, max_frames: int | None = None):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def _align(source, generated):
    import cv2
    import numpy as np

    length = min(len(source), len(generated))
    height, width = generated.shape[1:3]
    source = np.stack([
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in source[:length]
    ])
    return source, generated[:length]


def _pixel_metrics(source, generated, mask=None) -> dict[str, dict[str, Any]]:
    import numpy as np
    from skimage.metrics import structural_similarity

    source_f = source.astype(np.float32) / 255.0
    generated_f = generated.astype(np.float32) / 255.0
    if mask is None:
        valid = np.ones((*source_f.shape[:3], 1), dtype=np.float32)
        metric_prefix = "source_fullframe"
    else:
        valid = 1.0 - mask[..., None].astype(np.float32)
        metric_prefix = "source_unedited"
    valid_count = max(float(valid.sum() * 3), 1.0)
    mse = float((((source_f - generated_f) ** 2) * valid).sum() / valid_count)
    mae = float((abs(source_f - generated_f) * valid).sum() / valid_count)
    # Cap exact matches so the JSON remains standards-compliant and aggregable.
    psnr = 100.0 if mse == 0 else float(-10.0 * np.log10(mse))
    # SSIM is evaluated per frame. With a mask we report a conservative
    # masked luminance similarity because skimage has no masked SSIM API.
    if mask is None:
        ssim = float(np.mean([
            structural_similarity(a, b, channel_axis=2, data_range=1.0)
            for a, b in zip(source_f, generated_f)
        ]))
    else:
        similarity = 1.0 - abs(source_f - generated_f).mean(axis=3)
        denom = max(float(valid[..., 0].sum()), 1.0)
        ssim = float((similarity * valid[..., 0]).sum() / denom)
    if len(source_f) > 1:
        source_delta = source_f[1:] - source_f[:-1]
        generated_delta = generated_f[1:] - generated_f[:-1]
        temporal_delta_error = float(abs(source_delta - generated_delta).mean())
    else:
        temporal_delta_error = 0.0
    return {
        f"{metric_prefix}_psnr": {"value": psnr, "higher_is_better": True},
        f"{metric_prefix}_similarity": {"value": ssim, "higher_is_better": True},
        f"{metric_prefix}_mae": {"value": mae, "higher_is_better": False},
        "source_motion_delta_error": {
            "value": temporal_delta_error,
            "higher_is_better": False,
        },
    }


def _reference_metrics(reference, generated) -> dict[str, dict[str, Any]]:
    reference, generated = _align(reference, generated)
    metrics = _pixel_metrics(reference, generated)
    return {
        key.replace("source_fullframe", "reference"): value
        for key, value in metrics.items()
        if key != "source_motion_delta_error"
    }


def _clip_alignment(frames, prompt: str, model_name: str, pretrained: str):
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    # Uniformly cap the cost while retaining long-video coverage.
    indices = np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
    images = torch.stack([
        preprocess(Image.fromarray(frames[index])) for index in indices
    ]).to(device)
    text = tokenizer([prompt]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).mean().item()
    return {"value": float(score), "higher_is_better": True}


def _load_mask(path: str, target_frames, height: int, width: int):
    import cv2
    import numpy as np

    mask = _load_video(path, max_frames=target_frames)
    mask = np.stack([
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        for frame in mask
    ])
    if len(mask) < target_frames:
        mask = np.concatenate(
            [mask, np.repeat(mask[-1:], target_frames - len(mask), axis=0)]
        )
    return mask[:target_frames].mean(axis=3) >= 127.5


def evaluate_run(run_path: Path, args) -> bool:
    with open(run_path, encoding="utf-8") as handle:
        run = json.load(handle)
    video_path = Path(run["output_video"])
    if not video_path.exists():
        print(f"SKIP missing output: {video_path}")
        return False
    generated = _load_video(str(video_path))
    source = _load_video(run["case"]["source_path"])
    source, generated = _align(source, generated)
    mask = None
    if run["case"].get("mask_path"):
        mask = _load_mask(
            run["case"]["mask_path"], len(generated),
            generated.shape[1], generated.shape[2],
        )
    metrics = _pixel_metrics(source, generated, mask=mask)
    if run["case"].get("reference_path"):
        reference = _load_video(run["case"]["reference_path"])
        metrics.update(_reference_metrics(reference, generated))
    if args.clip:
        metrics["clip_instruction_alignment"] = _clip_alignment(
            generated, run["case"]["instruction"],
            args.clip_model, args.clip_pretrained,
        )
    trace_path = run_path.parent / "history_trace.json"
    runtime = {}
    if trace_path.exists():
        with open(trace_path, encoding="utf-8") as handle:
            runtime = json.load(handle).get("runtime", {})
        for name, value in runtime.items():
            if value is not None:
                metrics[f"runtime_{name}"] = {
                    "value": float(value), "higher_is_better": False,
                }
    result = {
        "run_id": run["run_id"],
        "case_id": run["case_id"],
        "variant_id": run["variant_id"],
        "seed": run["seed"],
        "git_commit": run.get("git_commit"),
        "frame_count": len(generated),
        "runtime": runtime,
        "metrics": metrics,
    }
    with open(run_path.parent / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    args = parser.parse_args()
    run_paths = sorted(Path(args.output_root).glob("runs/*/run.json"))
    completed = sum(evaluate_run(path, args) for path in run_paths)
    print(f"Evaluated {completed}/{len(run_paths)} runs")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
