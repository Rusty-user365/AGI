"""Materialize a deterministic split from the configured CGR corpus.

Example:
    python scripts/export_corpus.py --preset diffugpt \
        --out data/processed/diffugpt_train.jsonl

Exports document text and generator metadata. This is the exact synthetic
benchmark corpus, not a natural-language dataset or an independent sample of
human essays. Re-run with the same preset and seed to reproduce it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cgr.config import Config
from cgr.data.corpus import build_corpus
from cgr.data.grammar import (BODY_TEMPLATES, CONCLUSION_TEMPLATES,
                              INTRO_TEMPLATES, THESES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="diffugpt",
                    choices=["smoke", "dev", "paper", "diffugpt"])
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "heldout", "unseen", "all"])
    ap.add_argument("--n-docs", type=int, default=None,
                    help="override training document count")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()

    cfg = Config.preset(args.preset)
    if args.n_docs is not None:
        cfg.data.n_docs = args.n_docs
    if args.seed is not None:
        cfg.data.seed = args.seed
    tok, splits = build_corpus(cfg)
    names = list(splits) if args.split == "all" else [args.split]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for name in names:
            records = splits[name]
            counts[name] = len(records)
            for rec in records:
                row = {
                    "doc_id": rec.doc_id,
                    "split": name,
                    "thesis_id": rec.thesis_id,
                    "thesis": THESES[rec.thesis_id][0],
                    "template_combo": list(rec.combo),
                    "blocks": rec.blocks,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "preset": cfg.preset_name,
        "split": names,
        "counts": counts,
        "data_seed": cfg.data.seed,
        "tokenizer": tok.spec(),
        "template_counts": {
            "intro": len(INTRO_TEMPLATES),
            "body": len(BODY_TEMPLATES),
            "conclusion": len(CONCLUSION_TEMPLATES),
        },
        "n_theses": len(THESES),
        "n_heldout_theses": cfg.data.n_heldout_theses,
        "source": "deterministic CGR synthetic generator",
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "manifest": str(manifest_path),
                      "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
