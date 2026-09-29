#!/usr/bin/env python3
"""Convert LiveEdit's existing v2v JSON into experiment-manifest cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    parser.add_argument("--id-prefix", default="case")
    parser.add_argument(
        "--root-dir", default=".",
        help="Base for relative video paths (LiveEdit normally uses its launch cwd)",
    )
    args = parser.parse_args()
    source = Path(args.input_json).resolve()
    root_dir = Path(args.root_dir).resolve()
    with open(source, encoding="utf-8") as handle:
        items = json.load(handle)
    cases = []
    for index, item in enumerate(items):
        case = {
            "id": f"{args.id_prefix}-{index:04d}",
            "instruction": item["instruction"],
            "source_path": str((root_dir / item["source_path"]).resolve())
            if not Path(item["source_path"]).is_absolute()
            else item["source_path"],
        }
        if "edited_path" in item:
            reference = Path(item["edited_path"])
            case["reference_path"] = str(
                (root_dir / reference).resolve()
                if not reference.is_absolute() else reference
            )
        cases.append(case)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(cases, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cases)} cases to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
