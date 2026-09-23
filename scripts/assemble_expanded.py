#!/usr/bin/env python3
"""Reassemble data/{domain}.txt from judged journals (no API calls).

The journals are the single provenance record: every judged group stores the
orig_task and orig_subs it was built from, so the human-written source examples
are recovered from the journals themselves and emitted in a deterministic
task-sorted order, followed by Jev-accepted synthetic examples.

Usage: python3 scripts/assemble_expanded.py [--threshold 0.84] [--out data]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expand_dataset import GEN, ROOT, assemble_accepted

DOUBLE_ARTICLE = re.compile(
    r"\b(a|an) (scalable|secure|automated|lightweight) (a|an) ", re.IGNORECASE)


def repair_task(task):
    return DOUBLE_ARTICLE.sub(r"\1 \2 ", task, count=1)


def source_examples(records):
    by_task = {}
    for r in records:
        if r.get("orig_task") and r.get("orig_subs"):
            by_task[r["orig_task"]] = r["orig_subs"]
    return [{"task": t, "subs": by_task[t]} for t in sorted(by_task)]


def block(domain, task, subs):
    lines = [f"DOMAIN: {domain}", f"TASK: {task}"]
    lines += [f"SUB: {s}" for s in subs]
    return "\n".join(lines) + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.84)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for d in ["swe", "work", "creative"]:
        recs = [json.loads(l) for l in (GEN / f"{d}_judged.jsonl").open() if l.strip()]
        for r in recs:
            r["task"] = repair_task(r["task"])
        source = source_examples(recs)
        accepted = assemble_accepted(recs, args.threshold, 10 ** 9)

        text = "".join(block(d, ex["task"], ex["subs"]) for ex in source)
        text += "".join(block(d, ex["task"], ex["subs"]) for ex in accepted)
        out = out_dir / f"{d}.txt"
        out.write_text(text.rstrip("\n") + "\n")

        n = len(source) + len(accepted)
        total += n
        print(f"{d}: {len(source)} source + {len(accepted)} synthetic = {n} -> {out}")
    print(f"total examples: {total}")


if __name__ == "__main__":
    main()
