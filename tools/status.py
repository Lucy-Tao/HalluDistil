#!/usr/bin/env python3
"""Stock-take of HalluDistil experiment state.

Reports completed points, points awaiting judging, recent job failures and
the current queue. Read-only: it inspects files and calls squeue, and never
submits, cancels or modifies anything.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

# Judged files carry one record per evaluation item. Anything short of the
# full count is a partial run, whatever the job's exit status said.
SHORT_FORM_COMPLETE = 500
LONG_FORM_COMPLETE = 100

# Checked before any error pattern, because a retry loop can print a
# traceback from an early attempt and still succeed later.
SUCCESS_MARKERS = [
    "All 3 steps done",
    "judging done",
    "SKIPPED, judge separately",
    "succeeded.",
    "Done. Wrote to",
    "Done. Output in",
    "===== [.*] Done",
]

FAILURE_PATTERNS = [
    (r"OutOfMemoryError", "OOM, another process held the card"),
    (r"GPU_GATE_FAIL", "gate declined, too little free memory or unhealthy card"),
    (r"doesn't support bf16/gpu", "CUDA saw zero devices, usually a faulty card"),
    (r"OfflineModeIsEnabled", "dataset not cached on this node"),
    (r"source: not found|conda: not found", "--wrap ran under /bin/sh"),
    (r"PermissionError", "lock or cache directory owned by another user"),
    (r"device-side assert|probability tensor contains", "corrupt cross-GPU copy"),
    (r"DUE TO TIME LIMIT", "hit the time limit"),
    (r"CANCELLED", "cancelled"),
    (r"^[A-Za-z_]*Error", "python error"),
]

LINES = [
    ("main", "", "distill_and_eval_v3.sh"),
    ("noskip", "_noskip", "distill_and_eval_noskip.sh"),
    ("filter", "_se_filter", "distill_and_eval_se.sh"),
    ("replace", "_se_replace", "distill_and_eval_se.sh"),
    ("raw", "_raw", "distill_and_eval_raw.sh"),
    ("qwen25", "_qwen25", "distill_and_eval_qwen25.sh"),
    ("olmo", "_olmo", "distill_and_eval_olmo.sh"),
]

BASELINE_DIRS = [
    ("Qwen3", "judged_data_seed44_deberta"),
    ("Qwen2.5", "judged_data_seed44_qwen25"),
    ("OLMo 2", "judged_data_seed44_olmo"),
    ("Llama 3.1", "judged_data_seed44_llama_deberta"),
]


def line_count(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return -1


def tag_of(path):
    """Reduce a gen or judged filename to a comparable experiment tag.

    The two naming schemes differ in three places, so reconstructing one
    name from the other is error-prone. Everything up to and including
    seed44_ is dropped instead, which leaves the same tag on both sides.
    """
    base = os.path.basename(path)
    base = re.sub(r"\.jsonl$", "", base)
    m = re.search(r"seed44_(.*)$", base)
    if m:
        return m.group(1)
    # Baseline files have no seed44 segment, e.g. judged_simpleqa_Qwen3-32B_strict
    return re.sub(r"^(gen|judged)_simpleqa_(simpleqa_)?", "", base)


def jsonls(d):
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl")
    )


def section_complete(proj):
    print("=" * 62)
    print("COMPLETE  (judged, full length)")
    print("=" * 62)
    total = 0
    for name, suffix, _ in LINES:
        d = os.path.join(proj, f"judged_data_distilled_seed44{suffix}")
        files = jsonls(d)
        done = [f for f in files if line_count(f) == SHORT_FORM_COMPLETE]
        short = [f for f in files if 0 < line_count(f) < SHORT_FORM_COMPLETE]
        if not files:
            continue
        total += len(done)
        print(f"\n{name}  ({len(done)} of {len(files)})")
        for f in done:
            print(f"    {tag_of(f)}")
        for f in short:
            print(f"    {tag_of(f)}   PARTIAL {line_count(f)}/{SHORT_FORM_COMPLETE}")

    print(f"\nbaselines and teachers")
    for label, d in BASELINE_DIRS:
        files = jsonls(os.path.join(proj, d))
        done = [f for f in files if line_count(f) == SHORT_FORM_COMPLETE]
        total += len(done)
        for f in done:
            print(f"    {label:10s} {tag_of(f)}")

    lf = os.path.join(proj, "gen_longform_data")
    if os.path.isdir(lf):
        fs = [f for f in jsonls(lf) if os.path.basename(f).startswith("factscore_")]
        ent = [f for f in jsonls(lf) if os.path.basename(f).startswith("entropy_")]
        print(f"\nlong-form")
        print(f"    factscore judged   {len(fs)} sets")
        for f in sorted(ent):
            n = line_count(f)
            print(f"    claim entropy      {os.path.basename(f)[8:-6]:32s} {n} claims")

    print(f"\nshort-form points complete: {total}")


def section_pending(proj):
    print("=" * 62)
    print("GENERATED, AWAITING JUDGING")
    print("=" * 62)
    any_found = False
    for name, suffix, _ in LINES:
        gd = os.path.join(proj, f"gen_data_distilled_seed44{suffix}")
        jd = os.path.join(proj, f"judged_data_distilled_seed44{suffix}")
        gen = {tag_of(f): f for f in jsonls(gd)
               if line_count(f) == SHORT_FORM_COMPLETE}
        judged = {tag_of(f) for f in jsonls(jd)
                  if line_count(f) == SHORT_FORM_COMPLETE}
        missing = sorted(set(gen) - judged)
        if missing:
            any_found = True
            print(f"\n{name}  ({len(missing)})")
            for t in missing:
                print(f"    {t}")
                print(f"      {os.path.relpath(gen[t], proj)}")

    for label, jdir in BASELINE_DIRS:
        gdir = jdir.replace("judged_data", "gen_data").replace("_deberta", "")
        gd, jd = os.path.join(proj, gdir), os.path.join(proj, jdir)
        gen = {tag_of(f): f for f in jsonls(gd)
               if line_count(f) == SHORT_FORM_COMPLETE}
        judged = {tag_of(f) for f in jsonls(jd)
                  if line_count(f) == SHORT_FORM_COMPLETE}
        missing = sorted(set(gen) - judged)
        if missing:
            any_found = True
            print(f"\n{label} baseline  ({len(missing)})")
            for t in missing:
                print(f"    {t}")
                print(f"      {os.path.relpath(gen[t], proj)}")

    if not any_found:
        print("\n    nothing waiting")
    else:
        print("\nSubmit these with judge_se.sh. Judging needs about 66GB, so")
        print("use --array=1-N%1 and keep it off nodes running distillation.")


def section_failures(proj, limit):
    print("=" * 62)
    print(f"OUTCOMES  (most recent {limit} logs)")
    print("=" * 62)
    logdir = os.path.join(proj, "logs")
    if not os.path.isdir(logdir):
        print("    no logs directory")
        return
    outs = [os.path.join(logdir, f) for f in os.listdir(logdir)
            if f.endswith(".out")]
    outs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    succ = re.compile("|".join(SUCCESS_MARKERS))
    counts = defaultdict(int)
    failures = []

    for out in outs[:limit]:
        err = out[:-4] + ".err"
        # --wrap jobs without --error merge stderr into .out, so read both.
        text = ""
        for p in (out, err):
            try:
                with open(p, "r", errors="replace") as f:
                    text += f.read()
            except OSError:
                pass
        name = os.path.basename(out)[:-4]
        if succ.search(text):
            counts["done"] += 1
            continue
        reason = None
        for pat, desc in FAILURE_PATTERNS:
            if re.search(pat, text, re.M):
                reason = desc
                break
        if reason:
            counts["failed"] += 1
            node = ""
            m = re.search(r"host=(oat\d+)", text)
            if m:
                node = m.group(1)
            failures.append((name, node, reason))
        else:
            counts["running or unknown"] += 1

    for name, node, reason in failures:
        clean = "".join(c for c in name if c.isprintable())
        print(f"    {clean:34s} {node:6s} {reason}")
    if not failures:
        print("    no failures in this window")
    print("\n    " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


def section_history(proj, days, limit):
    """Reconstruct what was submitted and how it ended.

    Nothing is recorded at submission time. The manifest already carries
    job id, stage, parameters and node because the submission scripts
    append to it, and the logs carry the outcome, so the two are joined
    here rather than maintaining a separate ledger that could drift.
    """
    import datetime
    print("=" * 62)
    print(f"HISTORY  (last {days} days, most recent {limit})")
    print("=" * 62)

    man = os.path.join(proj, "logs", "experiment_manifest.log")
    if not os.path.isfile(man):
        print("    no manifest")
        return

    cutoff = datetime.datetime.now().astimezone() - datetime.timedelta(days=days)
    jobs = {}
    with open(man, "r", errors="replace") as f:
        for raw in f:
            parts = [x.strip() for x in raw.strip().split("|")]
            if len(parts) < 3:
                continue
            try:
                when = datetime.datetime.fromisoformat(parts[0])
            except ValueError:
                continue
            if when < cutoff:
                continue
            fields = {}
            for p2 in parts[1:]:
                if "=" in p2:
                    k, _, v = p2.partition("=")
                    fields[k.strip()] = v.strip()
            jid = fields.get("job")
            if not jid or jid == "none":
                continue
            rec = jobs.setdefault(jid, {"first": when, "last": when,
                                        "stages": [], "node": "", "desc": ""})
            rec["last"] = when
            st = fields.get("stage", "")
            if st:
                rec["stages"].append(st)
            if fields.get("node"):
                rec["node"] = fields["node"].split(".")[0]
            # Keep the most descriptive parameter set seen for this job.
            bits = [f"{k}={fields[k]}" for k in
                    ("style", "prompt_style", "level", "mode", "raw_samples",
                     "epochs", "tag", "model") if k in fields]
            if bits and len(" ".join(bits)) > len(rec["desc"]):
                rec["desc"] = " ".join(bits)

    # Outcome comes from the logs, using the same rule as the failures
    # section: success markers first, because a retry loop can print a
    # traceback and still finish.
    logdir = os.path.join(proj, "logs")
    by_jid = defaultdict(list)
    if os.path.isdir(logdir):
        for f in os.listdir(logdir):
            if not (f.endswith(".out") or f.endswith(".err")):
                continue
            m = re.search(r"_(\d+)(?:_\d+)?\.(out|err)$", f)
            if m:
                by_jid[m.group(1)].append(os.path.join(logdir, f))

    succ = re.compile("|".join(SUCCESS_MARKERS))

    # Array tasks complicate this. The manifest records SLURM_JOB_ID,
    # which for an array task is the individual task id, while the log
    # file is named after the parent array id plus a task index. Exact
    # matching therefore misses them, so fall back to searching log
    # contents for the job id the manifest recorded.
    _content_cache = {}

    def logs_mentioning(jid):
        hits = []
        if not os.path.isdir(logdir):
            return hits
        for f in sorted(os.listdir(logdir), reverse=True)[:400]:
            if not f.endswith(".out"):
                continue
            fp = os.path.join(logdir, f)
            if fp not in _content_cache:
                try:
                    with open(fp, "r", errors="replace") as fh:
                        _content_cache[fp] = fh.read(200000)
                except OSError:
                    _content_cache[fp] = ""
            if f"job={jid}" in _content_cache[fp]:
                hits.append(fp)
                err = fp[:-4] + ".err"
                if os.path.isfile(err):
                    hits.append(err)
                break
        return hits

    def outcome(jid):
        text = ""
        paths = by_jid.get(jid) or logs_mentioning(jid)
        for p3 in paths:
            try:
                with open(p3, "r", errors="replace") as fh:
                    text += fh.read()
            except OSError:
                pass
        if not text:
            # Array tasks record their own SLURM_JOB_ID in the manifest
            # while the log is named after the parent array id, so the
            # two do not line up. Completion for these is better read
            # from the judged file counts than from here.
            return "array task, see COMPLETE"
        if succ.search(text):
            return "done"
        for pat, desc in FAILURE_PATTERNS:
            if re.search(pat, text, re.M):
                return desc
        return "running or unknown"

    rows = sorted(jobs.items(), key=lambda kv: kv[1]["first"], reverse=True)
    print(f"    {'JOBID':<9} {'WHEN':<12} {'NODE':<7} {'OUTCOME':<34} PARAMS")
    for jid, r in rows[:limit]:
        stage = r["stages"][0] if r["stages"] else ""
        desc = r["desc"] or stage
        line = (f"    {jid:<9} {r['first'].strftime('%m-%d %H:%M'):<12} "
                f"{r['node']:<7} {outcome(jid):<34} {desc[:60]}")
        print("".join(c for c in line if c.isprintable()))
    if len(rows) > limit:
        print(f"\n    {len(rows) - limit} older entries hidden, raise --history-limit")


def section_queue():
    print("=" * 62)
    print("QUEUE")
    print("=" * 62)
    user = os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-o", "%.14i %.10T %.10M %.18j %R"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        print(f"    squeue unavailable: {e}")
        return
    print(out.rstrip() or "    empty")
    print("\n    An array shown as 113464_[4-12%2] is one line for many tasks.")
    print("    JobArrayTaskLimit means the concurrency cap is working, not a fault.")
    print("    ReqNodeNotAvail means the request cannot be met now, not that a")
    print("    node is broken. Check real capacity with:")
    print('      sinfo -p msc -N -O "NodeList:9,Gres:16,GresUsed:26,StateLong:12"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.path.expanduser("~/SimpleQA"))
    ap.add_argument("--only", choices=["complete", "pending-judge",
                                       "failures", "queue", "history"])
    ap.add_argument("--history-days", type=int, default=3)
    ap.add_argument("--history-limit", type=int, default=40)
    ap.add_argument("--log-window", type=int, default=40,
                    help="how many recent logs to scan for outcomes")
    args = ap.parse_args()

    proj = os.path.expanduser(args.project)
    if not os.path.isdir(proj):
        sys.exit(f"project directory not found: {proj}")

    want = args.only
    if want in (None, "complete"):
        section_complete(proj)
        print()
    if want in (None, "pending-judge"):
        section_pending(proj)
        print()
    if want in (None, "failures"):
        section_failures(proj, args.log_window)
        print()
    if want == "history":
        section_history(proj, args.history_days, args.history_limit)
        print()
    if want in (None, "queue"):
        section_queue()


if __name__ == "__main__":
    main()
