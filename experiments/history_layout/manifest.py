"""Manifest validation and deterministic run expansion (stdlib only)."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ExpandedRun:
    run_id: str
    case: dict[str, Any]
    variant_id: str
    policy: dict[str, Any]
    inference_overrides: dict[str, Any]
    seed: int


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("version") != 1:
        raise ValueError("manifest.version must be 1")
    for key in ("dataset", "inference", "variants", "seeds", "output_root"):
        if key not in manifest:
            raise ValueError(f"manifest is missing required field {key!r}")
    case_ids = [case.get("id") for case in manifest["dataset"]]
    if any(not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("dataset case ids must be present and unique")
    for case in manifest["dataset"]:
        for key in ("source_path", "instruction"):
            if key not in case:
                raise ValueError(f"case {case['id']!r} is missing {key!r}")
    variant_ids = [variant.get("id") for variant in manifest["variants"]]
    if any(not value for value in variant_ids) or len(set(variant_ids)) != len(variant_ids):
        raise ValueError("variant ids must be present and unique")
    baseline = manifest.get("baseline_variant")
    if baseline is not None and baseline not in variant_ids:
        raise ValueError(f"baseline_variant {baseline!r} is not a variant id")


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _variant_grid(variant: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    sweep = variant.get("sweep", {})
    if not sweep:
        yield copy.deepcopy(dict(variant))
        return
    keys = sorted(sweep)
    for values in itertools.product(*(sweep[key] for key in keys)):
        expanded = copy.deepcopy(dict(variant))
        expanded.pop("sweep", None)
        suffix = []
        for key, value in zip(keys, values):
            _set_dotted(expanded, key, value)
            suffix.append(f"{key.split('.')[-1]}-{value}")
        expanded["id"] = f"{variant['id']}__{'__'.join(suffix)}"
        yield expanded


def _materialize_case_policy(
    policy: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(policy))
    planner = result.get("planner", {})
    if planner.get("type") == "scene_contrast" and "scene_by_chunk" not in planner:
        if "scene_by_chunk" not in case:
            raise ValueError(
                f"case {case['id']!r} needs scene_by_chunk for scene_contrast"
            )
        planner["scene_by_chunk"] = case["scene_by_chunk"]
    return result


def expand_runs(manifest: Mapping[str, Any]) -> list[ExpandedRun]:
    runs: list[ExpandedRun] = []
    for case, variant, seed in itertools.product(
        manifest["dataset"],
        [item for value in manifest["variants"] for item in _variant_grid(value)],
        manifest["seeds"],
    ):
        policy = _materialize_case_policy(variant["policy"], case)
        identity = {
            "case": case["id"],
            "variant": variant["id"],
            "seed": int(seed),
            "policy": policy,
            "inference_overrides": variant.get("inference_overrides", {}),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        run_id = f"{case['id']}__{variant['id']}__seed-{seed}__{digest}"
        runs.append(
            ExpandedRun(
                run_id=run_id,
                case=copy.deepcopy(case),
                variant_id=variant["id"],
                policy=policy,
                inference_overrides=copy.deepcopy(
                    variant.get("inference_overrides", {})
                ),
                seed=int(seed),
            )
        )
    return runs


def resolve_path(value: str, manifest_path: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())
