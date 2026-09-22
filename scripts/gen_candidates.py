#!/usr/bin/env python3
"""Synthesize candidate (task, subtask) examples by fragment recombination.

Reads data/{domain}.txt, mines task/subtask fragments, and emits per-domain
candidate groups. Each group is one novel task plus ~30 candidate subtasks
labeled by synthesis kind, including control groups (same-domain verbatim
subtasks = expected positives, wrong-domain subtasks = expected negatives)
used to calibrate the Jev Noul threshold.

Output: data/gen/candidates_{domain}.jsonl
  {"task": str, "kind": str, "candidates": [{"kind": str, "text": str}, ...]}

Usage: python3 scripts/gen_candidates.py [per-domain task groups, default 15]
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "gen"

DOMAINS = ["swe", "work", "creative"]

AUDIENCES = [
    "a small startup", "a mid-size nonprofit", "a distributed remote team",
    "a regulated healthcare provider", "a university department",
    "a bootstrapped solo founder", "an enterprise compliance team",
    "a local community organization", "a fast-growing e-commerce company",
    "a government agency",
]

TECH = {
    "swe": ["Kubernetes", "Rust", "Django", "React", "Terraform", "GraphQL",
            "SQLite", "Docker", "serverless functions", "a monorepo"],
    "work": ["Asana", "Salesforce", "Slack", "a shared calendar", "Notion",
             "a weekly newsletter", "a CRM export", "a kanban board",
             "quarterly OKRs", "a customer survey"],
    "creative": ["watercolor", "a podcast format", "a zine", "ink and charcoal",
                 "a weekly short-story series", "collage", "a photo essay",
                 "a radio drama script", "mixed media", "a serialized newsletter"],
}


def parse_domain(path):
    tasks, subs = [], []
    for block in path.read_text().split("\n\n"):
        task = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("TASK:"):
                task = line[5:].strip()
            elif line.startswith("SUB:"):
                subs.append(line[4:].strip())
        if task:
            tasks.append(task)
    return tasks, subs


SUB_OPS = [
    ("same_verbatim", 8),
    ("splice", 14),
    ("tech_tail", 4),
    ("wrong_domain", 4),
]


def sub_opener(sub):
    words = sub.split()
    if len(words) <= 2:
        return words[0] if words else sub
    return " ".join(words[:2]) if words[1].endswith("e") is False and words[0] in (
        "Configure", "Define", "Prepare", "Validate", "Organize") else words[0]


def make_subtask_candidates(rng, own_subs, other_subs, tech):
    cands, seen = [], set()

    def push(kind, text):
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            cands.append({"kind": kind, "text": text})

    for kind, n in SUB_OPS:
        for _ in range(n):
            if kind == "same_verbatim":
                push(kind, rng.choice(own_subs))
            elif kind == "splice":
                a = rng.choice(own_subs)
                b = rng.choice(own_subs)
                op = sub_opener(a)
                b_words = b.split()
                if op.lower() == b_words[0].lower() or len(b_words) < 3:
                    push(kind, b)
                else:
                    push(kind, f"{op} {' '.join(b_words[1:])}")
            elif kind == "tech_tail":
                push(kind, f"{rng.choice(own_subs)} using {rng.choice(tech)}")
            elif kind == "wrong_domain":
                push(kind, rng.choice(other_subs))
    return cands


def make_task(rng, own_tasks, other_tech_tasks, domain):
    base = rng.choice(own_tasks)
    op = rng.choice(["audience", "tech", "splice"])
    if op == "audience":
        return "audience", f"{base} for {rng.choice(AUDIENCES)}"
    if op == "tech":
        return "tech", f"{base} using {rng.choice(TECH[domain])}"
    other_base = rng.choice(other_tech_tasks)
    verb = base.split()[0]
    rest = " ".join(other_base.split()[1:])
    return "splice", f"{verb} {rest}"


def main():
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    rng = random.Random(1234)
    OUT.mkdir(parents=True, exist_ok=True)

    parsed = {d: parse_domain(DATA / f"{d}.txt") for d in DOMAINS}

    total = 0
    for d in DOMAINS:
        own_tasks, own_subs = parsed[d]
        others = [parsed[o][1] for o in DOMAINS if o != d]
        other_subs = [s for subs in others for s in subs]
        other_tech_tasks = [t for o in DOMAINS if o != d for t in parsed[o][0]]

        orig_tasks = set(own_tasks)
        groups = []
        attempts = 0
        while len(groups) < n_tasks and attempts < n_tasks * 20:
            attempts += 1
            kind, task = make_task(rng, own_tasks, other_tech_tasks, d)
            if task in orig_tasks:
                continue
            cands = make_subtask_candidates(rng, own_subs, other_subs, TECH[d])
            if len(cands) < 20:
                continue
            groups.append({"task": task, "kind": kind, "candidates": cands})

        path = OUT / f"candidates_{d}.jsonl"
        with path.open("w") as f:
            for g in groups:
                f.write(json.dumps({"domain": d, **g}) + "\n")
        total += len(groups)
        print(f"{d}: {len(groups)} candidate tasks, "
              f"{sum(len(g['candidates']) for g in groups)} candidate subtasks -> {path}")
    print(f"total: {total} groups")


if __name__ == "__main__":
    main()
