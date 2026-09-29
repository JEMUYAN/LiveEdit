#!/usr/bin/env python3
"""Aggregate metrics and compute paired bootstrap differences vs baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _bootstrap_ci(values, samples=5000, seed=0):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--minimum-effect", action="append", default=[], metavar="METRIC=VALUE",
        help="Predeclared practical threshold; may be repeated",
    )
    args = parser.parse_args()
    minimum_effect = {}
    for item in args.minimum_effect:
        metric, value = item.split("=", 1)
        minimum_effect[metric] = float(value)
    root = Path(args.output_root)
    records = []
    for path in sorted(root.glob("runs/*/metrics.json")):
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    if not records:
        raise SystemExit("No metrics.json files found")

    flat = []
    by_pair = {}
    for record in records:
        for metric, payload in record["metrics"].items():
            row = {
                "run_id": record["run_id"], "case_id": record["case_id"],
                "variant_id": record["variant_id"], "seed": record["seed"],
                "metric": metric, "value": payload["value"],
                "higher_is_better": payload["higher_is_better"],
            }
            flat.append(row)
            by_pair[(row["case_id"], row["seed"], metric, row["variant_id"])] = row

    with open(root / "per_sample.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)

    groups = defaultdict(list)
    for row in flat:
        groups[(row["variant_id"], row["metric"])].append(float(row["value"]))
    aggregate = []
    for (variant, metric), values in sorted(groups.items()):
        aggregate.append({
            "variant_id": variant, "metric": metric, "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    with open(root / "aggregate.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)

    differences = defaultdict(list)
    directions = {}
    for row in flat:
        if row["variant_id"] == args.baseline:
            continue
        baseline = by_pair.get(
            (row["case_id"], row["seed"], row["metric"], args.baseline)
        )
        if baseline is None:
            continue
        raw_delta = float(row["value"]) - float(baseline["value"])
        favorable_delta = raw_delta if row["higher_is_better"] else -raw_delta
        differences[(row["variant_id"], row["metric"])].append(favorable_delta)
        directions[row["metric"]] = row["higher_is_better"]
    paired = []
    for (variant, metric), values in sorted(differences.items()):
        low, high = _bootstrap_ci(values, args.bootstrap_samples)
        paired.append({
            "variant_id": variant, "metric": metric, "n": len(values),
            "mean_favorable_delta": statistics.fmean(values),
            "ci95_low": low, "ci95_high": high,
            "statistically_directional": low > 0 or high < 0,
            "practically_directional": (
                (low > 0 or high < 0)
                and abs(statistics.fmean(values)) >= minimum_effect.get(metric, 0.0)
            ),
            "minimum_effect": minimum_effect.get(metric, 0.0),
            "direction": "higher is better" if directions[metric] else "lower is better",
        })
    paired_fields = [
        "variant_id", "metric", "n", "mean_favorable_delta", "ci95_low",
        "ci95_high", "statistically_directional", "practically_directional",
        "minimum_effect", "direction",
    ]
    with open(root / "paired_deltas.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired)
    with open(root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"aggregate": aggregate, "paired": paired}, handle, indent=2)

    lines = [
        "# History-layout experiment report", "",
        f"Paired baseline: `{args.baseline}`. Positive deltas favor the candidate.", "",
        "| Variant | Metric | n | Mean favorable delta | 95% bootstrap CI | Statistical | Practical |",
        "|---|---|---:|---:|---:|:---:|:---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['variant_id']} | {row['metric']} | {row['n']} | "
            f"{row['mean_favorable_delta']:.6g} | "
            f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | "
            f"{'yes' if row['statistically_directional'] else 'no'} | "
            f"{'yes' if row['practically_directional'] else 'no'} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote aggregate and paired reports under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
