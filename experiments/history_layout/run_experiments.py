#!/usr/bin/env python3
"""Expand a history-layout manifest and run one isolated process per sample."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from manifest import expand_runs, load_manifest, resolve_path


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_command(
    repo_root: Path,
    manifest_path: Path,
    manifest,
    run,
    run_dir: Path,
) -> tuple[list[str], Path]:
    inference = manifest["inference"]
    data_path = run_dir / "input.json"
    policy_path = run_dir / "policy.json"
    trace_path = run_dir / "history_trace.json"
    output_dir = run_dir / "videos"
    source_path = resolve_path(run.case["source_path"], manifest_path)
    item = {"source_path": source_path, "instruction": run.case["instruction"]}
    if "reference_path" in run.case:
        item["edited_path"] = resolve_path(run.case["reference_path"], manifest_path)
    _write_json(data_path, [item])
    _write_json(policy_path, run.policy)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        inference.get("python", sys.executable),
        str(repo_root / "inference-mm.py"),
        "--config_path", resolve_path(inference["config_path"], manifest_path),
        "--checkpoint_path", resolve_path(inference["checkpoint_path"], manifest_path),
        "--data_path", str(data_path),
        "--output_folder", str(output_dir),
        "--task", "v2v",
        "--num_output_frames", str(inference["num_output_frames"]),
        "--inference_num_steps", str(inference.get("inference_num_steps", 4)),
        "--seed", str(run.seed),
        "--num_samples", "1",
        "--save_with_index",
        "--prefix", f"{run.run_id}-",
        "--history_policy_config", str(policy_path),
        "--history_trace_path", str(trace_path),
        "--return_generation_time",
    ]
    merged_args = dict(inference.get("extra_args", {}))
    merged_args.update(run.inference_overrides)
    for key, value in merged_args.items():
        flag = "--" + key.replace("_", "-")
        # inference-mm currently uses underscore flags; preserve explicit keys.
        if "_" in key:
            flag = "--" + key
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    model = "ema" if "--use_ema" in command else "regular"
    expected_video = output_dir / f"{run.run_id}-0-0_{model}.mp4"
    return command, expected_video


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", action="append", default=[],
                        help="Run-id substring filter; may be repeated")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(resolve_path(manifest["output_root"], manifest_path))
    runs = expand_runs(manifest)
    if args.only:
        runs = [run for run in runs if any(term in run.run_id for term in args.only)]
    print(f"Validated {len(runs)} runs; output_root={output_root}")

    failures = 0
    for run in runs:
        run_dir = output_root / "runs" / run.run_id
        command, expected_video = build_command(
            repo_root, manifest_path, manifest, run, run_dir
        )
        record_path = run_dir / "run.json"
        resolved_case = dict(run.case)
        for path_key in ("source_path", "reference_path", "mask_path"):
            if resolved_case.get(path_key):
                resolved_case[path_key] = resolve_path(
                    resolved_case[path_key], manifest_path
                )
        record = {
            "run_id": run.run_id,
            "case_id": run.case["id"],
            "variant_id": run.variant_id,
            "seed": run.seed,
            "case": resolved_case,
            "policy": run.policy,
            "inference_overrides": run.inference_overrides,
            "git_commit": _git_commit(repo_root),
            "command": command,
            "output_video": str(expected_video),
            "status": "planned",
        }
        if args.dry_run:
            _write_json(record_path, record)
            continue
        if args.resume and expected_video.exists():
            record["status"] = "skipped_existing"
            _write_json(record_path, record)
            continue
        started = time.time()
        record["status"] = "running"
        _write_json(record_path, record)
        with open(run_dir / "inference.log", "w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT,
                check=False, env=os.environ.copy(),
            )
        record["elapsed_seconds"] = time.time() - started
        record["return_code"] = result.returncode
        record["status"] = "completed" if result.returncode == 0 else "failed"
        _write_json(record_path, record)
        if result.returncode != 0:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
