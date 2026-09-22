#!/usr/bin/env python3
"""Jev gate: judge synthesized candidate task/subtask examples via TypeSafe API.

One request per candidate task group: a Noul per candidate subtask plus a
task-realism Noul, all batched into a single call (parallel questions share
state, which is the cheap request shape per TypeSafe docs).

Modes:
  smoke      one request, print the raw answer
  calibrate  judge all data/gen/candidates_{domain}.jsonl groups, write
             data/gen/judged_calib.jsonl + data/gen/calibration_report.txt

Usage:
  python3 scripts/jev_gate.py smoke
  python3 scripts/jev_gate.py calibrate [groups-per-domain]
"""

import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "gen"
KEY_PATH = Path.home() / "Documents" / "jevapi.txt"
API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
DOMAINS = ["swe", "work", "creative"]
WORKERS = 8
THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]


def api_key():
    return KEY_PATH.read_text().strip()


def ask(key, state, questions, tries=5):
    body = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    delay = 1.0
    for attempt in range(tries):
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except urllib.error.URLError:
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def build_request(group):
    state = {
        "domain": group["domain"],
        "task": group["task"],
        "candidates": {f"c{i}": c["text"] for i, c in enumerate(group["candidates"])},
    }
    questions = {
        "task_ok": {
            "type": "noul",
            "instructions": "Is `task` a realistic, coherent professional task in the `domain` domain?",
            "criteria": {
                "true": "A task a real professional or team could plausibly be asked to do",
                "false": "Nonsensical, incoherent, mixes unrelated activities, or no professional would frame work this way",
            },
        }
    }
    for i, c in enumerate(group["candidates"]):
        cid = f"c{i}"
        questions[cid] = {
            "type": "noul",
            "instructions": (
                f"Considering the `domain` task `task`, does `candidates.{cid}` "
                "describe a coherent, actionable subtask step toward completing that task?"
            ),
            "criteria": {
                "true": "A specific, actionable step a competent professional would plausibly include when executing the task",
                "false": "Irrelevant to the task, incoherent, too vague to act on, a restatement of the task itself, or work belonging to an unrelated domain",
            },
        }
    return state, questions


def judge_group(key, group):
    state, questions = build_request(group)
    resp = ask(key, state, questions)
    judged = []
    for i, c in enumerate(group["candidates"]):
        cid = f"c{i}"
        judged.append({**c, "noul": resp["answers"][cid]["noul"]})
    return {
        "domain": group["domain"],
        "task": group["task"],
        "task_kind": group["kind"],
        "task_ok": resp["answers"]["task_ok"]["noul"],
        "candidates": judged,
        "usage": resp.get("usage", {}),
    }


def smoke():
    key = api_key()
    group = next(
        json.loads(l) for l in (GEN / "candidates_swe.jsonl").open() if l.strip()
    )
    state, questions = build_request(group)
    questions = dict(list(questions.items())[:3])
    resp = ask(key, state, questions)
    print(json.dumps(resp, indent=2))


def calibrate(limit_per_domain=None):
    key = api_key()
    groups = []
    for d in DOMAINS:
        path = GEN / f"candidates_{d}.jsonl"
        rows = [json.loads(l) for l in path.open() if l.strip()]
        if limit_per_domain:
            rows = rows[:limit_per_domain]
        groups.extend(rows)
    print(f"judging {len(groups)} groups ({sum(len(g['candidates']) for g in groups)} candidate subtasks)...")

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(judge_group, key, g): g for g in groups}
        done = 0
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"FAILED group: {futs[fut]['task'][:60]!r}: {e}")
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(groups)}")

    out = GEN / "judged_calib.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    total_in = sum(r["usage"].get("input_tokens", 0) for r in results)
    total_out = sum(r["usage"].get("output_tokens", 0) for r in results)
    print(f"wrote {len(results)} judged groups -> {out}")
    print(f"usage: {total_in} input tokens, {total_out} output tokens")
    write_report(results)


def write_report(results):
    lines = ["=== Jev calibration report ===", ""]
    by_kind = {}
    for r in results:
        for c in r["candidates"]:
            by_kind.setdefault(c["kind"], []).append(c["noul"])

    lines.append("kind            n     mean   | kept at threshold")
    lines.append("                             | " + "  ".join(f">={t:.2f}" for t in THRESHOLDS))
    for kind in ["same_verbatim", "splice", "tech_tail", "wrong_domain"]:
        vals = by_kind.get(kind, [])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        kept = "  ".join(f"{sum(v >= t for v in vals):5d}" for t in THRESHOLDS)
        lines.append(f"{kind:15s} {len(vals):4d}  {mean:6.3f}  | {kept}")

    task_ok = [r["task_ok"] for r in results]
    lines.append("")
    lines.append(f"task realism: n={len(task_ok)} mean={sum(task_ok)/len(task_ok):.3f} "
                 f"min={min(task_ok):.3f} tasks<0.5={sum(v < 0.5 for v in task_ok)}")

    spliced = sorted(
        (c for r in results for c in r["candidates"] if c["kind"] == "splice"),
        key=lambda c: -c["noul"])
    lines.append("")
    lines.append("top-5 spliced candidates (novel combinations):")
    for c in spliced[:5]:
        lines.append(f"  {c['noul']:.3f}  {c['text']}")
    lines.append("")
    lines.append("bottom-5 same-domain verbatim (should be rare):")
    verbatim = sorted(
        (c for r in results for c in r["candidates"] if c["kind"] == "same_verbatim"),
        key=lambda c: c["noul"])
    for c in verbatim[:5]:
        lines.append(f"  {c['noul']:.3f}  {c['text']}")
    lines.append("")
    lines.append("bottom-3 wrong-domain (sanity: all should be low):")
    wrong = sorted(
        (c for r in results for c in r["candidates"] if c["kind"] == "wrong_domain"),
        key=lambda c: c["noul"])
    for c in wrong[:3]:
        lines.append(f"  {c['noul']:.3f}  {c['text']}")

    bad_tasks = sorted((r for r in results if r["task_ok"] < 0.5),
                       key=lambda r: r["task_ok"])
    lines.append("")
    lines.append(f"tasks judged unrealistic (task_ok<0.5): {len(bad_tasks)}")
    for r in bad_tasks[:5]:
        lines.append(f"  {r['task_ok']:.3f}  [{r['task_kind']}] {r['task']}")

    out = GEN / "calibration_report.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"report -> {out}")


def parse_original_examples(domain, limit):
    examples = []
    for block in (ROOT / "data" / f"{domain}.txt").read_text().split("\n\n"):
        task, subs = None, []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("TASK:"):
                task = line[5:].strip()
            elif line.startswith("SUB:"):
                subs.append(line[4:].strip())
        if task and subs:
            examples.append({"task": task, "subs": subs})
    return examples[:limit]


def controls(limit_per_domain=10):
    key = api_key()
    rng = random.Random(7)
    groups = []
    for d in DOMAINS:
        others = [o for o in DOMAINS if o != d]
        for ex in parse_original_examples(d, limit_per_domain):
            cands = [{"kind": "original_pair", "text": s} for s in ex["subs"]]
            pool = []
            for o in others:
                for line in (ROOT / "data" / f"{o}.txt").read_text().split("\n\n"):
                    for line2 in line.splitlines():
                        if line2.strip().startswith("SUB:"):
                            pool.append(line2.strip()[4:].strip())
            for _ in range(4):
                cands.append({"kind": "wrong_domain", "text": rng.choice(pool)})
            groups.append({"domain": d, "task": ex["task"], "kind": "control",
                           "candidates": cands})

    print(f"judging {len(groups)} control groups...")
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(judge_group, key, g) for g in groups]
        for fut in as_completed(futs):
            results.append(fut.result())

    out = GEN / "judged_controls.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    lines = ["=== control experiment: original pairs vs wrong-domain ===", ""]
    for kind in ["original_pair", "wrong_domain"]:
        vals = [c["noul"] for r in results for c in r["candidates"] if c["kind"] == kind]
        mean = sum(vals) / len(vals)
        kept = "  ".join(f">={t:.2f}: {sum(v >= t for v in vals):4d}/{len(vals)}"
                         for t in THRESHOLDS)
        lines.append(f"{kind:14s} n={len(vals):4d} mean={mean:.3f}  {kept}")
    orig = sorted((c["noul"] for r in results for c in r["candidates"]
                   if c["kind"] == "original_pair"))
    lines.append(f"original_pair deciles: {[round(orig[int(i*len(orig)/10)],3) for i in range(10)]}")
    out2 = GEN / "controls_report.txt"
    out2.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if mode == "smoke":
        smoke()
    elif mode == "calibrate":
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
        calibrate(lim)
    elif mode == "controls":
        controls(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        sys.exit(f"unknown mode {mode}")
