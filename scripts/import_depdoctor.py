"""Download a small generated-only DepDoctor slice and normalize it to JSONL.

Requires the optional `datasets` dependency. The selected source classes are
synthetic, augmented, and noop (released under Apache-2.0 per the dataset
license notice). The real mined rows have per-repository licenses and are
excluded by default. DepDoctor supplies no unit tests, so `unit_test` remains
null and verification is explicitly marked unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_ID = "Abhisek987/depdoctor-dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
GENERATED_SOURCES = {"synthetic", "augmented", "noop"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--out", type=Path,
                        default=Path("data/external/depdoctor_sample.jsonl"))
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install the optional data dependency first: "
                         "pip install -r requirements-data.txt") from e

    rows = load_dataset(DATASET_ID, split="train", streaming=True)
    selected = []
    for row in rows:
        if row.get("source") not in GENERATED_SOURCES:
            continue
        selected.append({
            "id": f"depdoctor-{row.get('id', len(selected))}",
            "task": "code_repair",
            "buggy_code": row.get("broken_code", ""),
            "fixed_code": row.get("fixed_code", ""),
            "unit_test": None,
            "unit_test_status": "not_provided_by_source_dataset",
            "verification_status": "unverified",
            "api_diff": row.get("api_diff", ""),
            "package": row.get("package"),
            "error": row.get("error"),
            "source_class": row.get("source"),
            "source_dataset": DATASET_ID,
            "source_dataset_url": DATASET_URL,
            "source_license": "Apache-2.0 (generated source class)",
            "attribution": {"repo": row.get("repo"), "pr": row.get("pr"),
                            "commit": row.get("commit")},
        })
        if len(selected) >= args.limit:
            break
    if len(selected) < args.limit:
        raise RuntimeError(f"requested {args.limit} generated rows; found {len(selected)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "dataset": DATASET_ID, "dataset_url": DATASET_URL,
        "citation": "Behera (2026), https://doi.org/10.5281/zenodo.20787803",
        "selected_source_classes": sorted(GENERATED_SOURCES),
        "license_note": "Generated rows are Apache-2.0 per the source LICENSE.",
        "count": len(selected), "unit_tests": "not supplied; no test was fabricated",
        "output": args.out.as_posix(),
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
