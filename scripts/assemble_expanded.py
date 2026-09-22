#!/usr/bin/env python3
"""Reassemble data/expanded/{domain}.txt from judged journals (no API calls).

Applies text repairs to synthetic task phrasing (double-article collisions
from the qualifier variant op) and re-runs acceptance at the given threshold.

Usage: python3 scripts/assemble_expanded.py [--threshold 0.84]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expand_dataset import GEN, ROOT, assemble_accepted, load_originals

DOUBLE_ARTICLE = re.compile(
    r"\b(a|an) (scalable|secure|automated|lightweight) (a|an) ", re.IGNORECASE)


def repair_task(task):
    return DOUBLE_ARTICLE.sub(r"\1 \2 ", task, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.84)
    args = ap.parse_args()

    originals = load_originals()
    total = 0
    for d in ["swe", "work", "creative"]:
        journal = GEN / f"{d}_judged.jsonl"
        recs = [json.loads(l) for l in journal.open() if l.strip()]
        for r in recs:
            r["task"] = repair_task(r["task"])
        acc = assemble_accepted(recs, args.threshold, 10 ** 9)
        out = ROOT / "data" / "expanded" / f"{d}.txt"
        with out.open("w") as f:
            for ex in originals[d]:
                f.write(f"DOMAIN: {d}\nTASK: {ex['task']}\n")
                for s in ex["subs"]:
                    f.write(f"SUB: {s}\n")
                f.write("\n")
            for ex in acc:
                f.write(f"DOMAIN: {d}\nTASK: {ex['task']}\n")
                for s in ex["subs"]:
                    f.write(f"SUB: {s}\n")
                f.write("\n")
        n = len(originals[d]) + len(acc)
        total += n
        print(f"{d}: {len(originals[d])} original + {len(acc)} synthetic = {n}")
    print(f"total examples: {total}")


if __name__ == "__main__":
    main()
