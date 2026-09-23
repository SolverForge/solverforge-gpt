#!/usr/bin/env python3
"""Jev-gated topical dataset expansion for solverforge-gpt.

For each original task, build topical variants (audience/tech/qualifier tails,
family verb-splices) and candidate subtask lists drawn from the task's
token-overlap family (own subs, family subs, family splices, tailed variants).
Each group is judged in one batched TypeSafe request (Noul per candidate +
task-realism Noul). Accepted examples are assembled into DOMAIN/TASK/SUB text
under data/{domain}.txt, preserving original examples.

Canary wrong-domain candidates ride along in every group to monitor judge
drift; they are never emitted. Judged groups are journaled to
data/expand_work/{domain}_judged.jsonl and reused on rerun.

Usage:
  python3 scripts/expand_dataset.py [--target 5000] [--threshold 0.88]
                                    [--max-requests 40000] [--workers 12]
                                    [--dry N]
"""

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_candidates as gc
import jev_gate as jg

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "expand_work"
STOPWORDS = set("""a an the and or for to with from into onto of in on by using
your our their it its this that build make create set up add""".split())

PHASE0 = {"plan", "define", "research", "analyze", "audit", "survey", "gather",
          "choose", "identify", "scope", "design", "draft", "outline", "assess",
          "brainstorm", "select", "map", "evaluate", "collect", "decide"}
PHASE1 = {"create", "build", "implement", "develop", "write", "configure",
          "install", "integrate", "migrate", "establish", "produce", "record",
          "edit", "film", "shoot", "compose", "code", "construct", "prepare",
          "assemble", "instrument", "arrange", "organize", "schedule"}
PHASE2 = {"test", "validate", "verify", "review", "refine", "revise", "qa",
          "check", "rehearse", "proof", "debug"}
PHASE3 = {"launch", "deploy", "publish", "release", "document", "train",
          "promote", "measure", "monitor", "maintain", "ship", "distribute",
          "present", "hand"}

QUALIFIERS = ["scalable", "secure", "automated", "lightweight"]


def phase(sub):
    verb = sub.split()[0].lower().strip(",")
    if verb in PHASE0:
        return 0
    if verb in PHASE2:
        return 2
    if verb in PHASE3:
        return 3
    return 1 if verb in PHASE1 else 1.5


def content_words(text):
    return {w.strip(".,:;()").lower() for w in text.split()} - STOPWORDS


def load_originals():
    out = {}
    for d in jg.DOMAINS:
        out[d] = jg.parse_original_examples(d, 10 ** 6)
    return out


def build_families(examples):
    fam = {}
    cw = [(ex["task"], content_words(ex["task"]),
           [content_words(s) for s in ex["subs"]]) for ex in examples]
    for i, (t1, w1, _) in enumerate(cw):
        fam[t1] = [t2 for (t2, w2, _) in cw if t1 != t2 and len(w1 & w2) >= 1]
    return fam


def task_variants(rng, base, family_tasks, domain):
    v = []
    fam = family_tasks
    v.append(("base", base))
    for a in rng.sample(gc.AUDIENCES, 8):
        v.append(("audience", f"{base} for {a}"))
    for t in rng.sample(gc.TECH[domain], 5):
        v.append(("tech", f"{base} using {t}"))
    for q in QUALIFIERS:
        words = base.split()
        v.append(("qualifier", f"{words[0]} a {q} {' '.join(words[1:])}"))
    for _ in range(6):
        if not fam:
            break
        other = rng.choice(fam)
        ow = other.split()
        bw = base.split()
        if len(ow) > 2 and len(bw) > 1:
            v.append(("splice", f"{bw[0]} {' '.join(ow[1:])}"))
    out, seen = [], set()
    for kind, t in v:
        if t not in seen:
            seen.add(t)
            out.append((kind, t))
    return out


def subtask_candidates(rng, own_subs, family_subs, domain, n=20):
    cands, seen = [], set()

    def push(kind, text):
        text = " ".join(text.split())
        if text and text not in seen:
            seen.add(text)
            cands.append({"kind": kind, "text": text})

    n_own = max(4, n // 3)
    pool_own = list(own_subs)
    pool_fam = list(family_subs)
    for _ in range(n_own):
        src = rng.random()
        if src < 0.45 or not pool_fam:
            push("own", rng.choice(pool_own))
        else:
            push("family", rng.choice(pool_fam))
    for _ in range(max(4, n // 4)):
        a = rng.choice(pool_own + pool_fam)
        b = rng.choice(pool_own + pool_fam)
        aw, bw = a.split(), b.split()
        if aw[0].lower() == bw[0].lower() or len(bw) < 3:
            push("splice", b)
        else:
            push("splice", f"{aw[0]} {' '.join(bw[1:])}")
    for _ in range(max(2, n // 8)):
        push("tech_tail", f"{rng.choice(pool_own + pool_fam)} using {rng.choice(gc.TECH[domain])}")
    for _ in range(max(2, n // 10)):
        push("canary", rng.choice(CANARY_POOL[domain]))
    return cands[:n]


CANARY_POOL = {}


def build_canary_pool(all_examples):
    for d in jg.DOMAINS:
        others = [o for o in jg.DOMAINS if o != d]
        CANARY_POOL[d] = [s for o in others for ex in all_examples[o] for s in ex["subs"]]


def make_groups(rng, domain, all_examples, families):
    groups = []
    for ex in all_examples[domain]:
        own_subs = ex["subs"]
        family_subs = [s for t in families.get(ex["task"], [])
                       for e2 in [next((x for x in all_examples[domain] if x["task"] == t), None)]
                       if e2 for s in e2["subs"]]
        for kind, task in task_variants(rng, ex["task"], families.get(ex["task"], []), domain):
            cands = subtask_candidates(rng, own_subs, family_subs, domain)
            groups.append({"domain": domain, "task": task, "kind": kind,
                           "orig_task": ex["task"], "orig_subs": own_subs,
                           "candidates": cands})
    return groups


def group_key(g):
    h = hashlib.sha256()
    h.update(g["task"].encode())
    for c in g["candidates"]:
        h.update(c["text"].encode())
    return h.hexdigest()[:16]


def judge_all(key, groups, workers, max_requests, threshold, target, domain, log):
    journal = GEN / f"{domain}_judged.jsonl"
    done = {}
    if journal.exists():
        for line in journal.open():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    todo = [g for g in groups if group_key(g) not in done]
    log(f"{domain}: {len(done)} journaled, {len(todo)} to judge")
    accepted = assemble_accepted(done.values(), threshold, target)
    if len(accepted) >= target:
        return done

    fh = journal.open("a")
    n_req = 0
    hit_target = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for g in todo:
            if n_req >= max_requests:
                break
            futs[ex.submit(jg.judge_group, key, g)] = g
            n_req += 1
        for fut in as_completed(futs):
            g = futs[fut]
            try:
                r = fut.result()
                rec = {"key": group_key(g), "domain": g["domain"], "task": g["task"],
                       "kind": g["kind"], "orig_task": g["orig_task"],
                       "orig_subs": g["orig_subs"],
                       "task_ok": r["task_ok"],
                       "candidates": r["candidates"],
                       "usage": r.get("usage", {})}
                done[rec["key"]] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            except Exception as e:
                log(f"  FAIL {g['task'][:50]!r}: {e}")
            if len(done) % 50 == 0:
                acc = assemble_accepted(done.values(), threshold, target)
                log(f"  {domain}: {len(done)} judged, {len(acc)} accepted")
                if len(acc) >= target:
                    hit_target = True
                    break
        if hit_target:
            for f in futs:
                f.cancel()
    fh.close()
    return done


def assemble_accepted(records, threshold, target):
    examples, seen_tasks = [], set()
    for r in records:
        if r.get("task_ok") is not None and r["task_ok"] < 0.6:
            continue
        kept = []
        seen = set()
        for c in r["candidates"]:
            if c["kind"] == "canary" or c["noul"] < threshold:
                continue
            if c["text"] in seen:
                continue
            seen.add(c["text"])
            kept.append(c)
        if len(kept) < 3:
            continue
        if r["task"] == r["orig_task"] and all(c["kind"] == "own" for c in kept):
            continue
        kept.sort(key=lambda c: (phase(c["text"]), c["noul"] * -0.001))
        kept = kept[:6]
        sig = (r["task"], tuple(sorted(c["text"] for c in kept)))
        if sig in seen_tasks:
            continue
        seen_tasks.add(sig)
        examples.append({"task": r["task"], "subs": [c["text"] for c in kept],
                         "kind": r["kind"]})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.88)
    ap.add_argument("--max-requests", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry", type=int, default=0)
    args = ap.parse_args()

    GEN.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    key = jg.api_key()

    all_examples = load_originals()
    build_canary_pool(all_examples)
    families = {d: build_families(all_examples[d]) for d in jg.DOMAINS}

    stats = {}
    budget = args.max_requests
    for d in jg.DOMAINS:
        groups = make_groups(rng, d, all_examples, families[d])
        rng.shuffle(groups)
        target = args.target if not args.dry else args.dry
        target = min(target, int(len(groups) * 0.7))
        t0 = time.time()

        def log(msg):
            print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

        done = judge_all(key, groups, args.workers, budget, args.threshold,
                         target, d, log)
        n_req = sum(1 for _ in done)
        budget = max(0, budget - n_req)
        acc = assemble_accepted(done.values(), args.threshold, target)[:target]
        out = ROOT / "data" / f"{d}.txt"
        source = all_examples[d]
        source_sigs = {(ex["task"], tuple(sorted(ex["subs"]))) for ex in source}
        added = [ex for ex in acc
                 if (ex["task"], tuple(sorted(ex["subs"]))) not in source_sigs]
        with out.open("w") as f:
            for ex in source:
                f.write(f"DOMAIN: {d}\nTASK: {ex['task']}\n")
                for s in ex["subs"]:
                    f.write(f"SUB: {s}\n")
                f.write("\n")
            for ex in added:
                f.write(f"DOMAIN: {d}\nTASK: {ex['task']}\n")
                for s in ex["subs"]:
                    f.write(f"SUB: {s}\n")
                f.write("\n")
        canary = [c for r in done.values() for c in r["candidates"] if c["kind"] == "canary"]
        canary_pass = sum(1 for c in canary if c["noul"] >= args.threshold)
        stats[d] = {
            "groups_judged": len(done),
            "accepted": len(acc),
            "canary_n": len(canary),
            "canary_false_pass": canary_pass,
            "tokens_in": sum(r["usage"].get("input_tokens", 0) for r in done.values()),
            "tokens_out": sum(r["usage"].get("output_tokens", 0) for r in done.values()),
        }
        log(f"{d}: wrote {len(all_examples[d])} original + {len(acc)} synthetic -> {out}")
        if budget == 0:
            log("request budget exhausted, stopping")
            break

    (GEN / "expansion_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
