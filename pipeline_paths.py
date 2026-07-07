"""
pipeline_paths.py — small helper for run_full_experiment.sh.

Resolves the SAME auto-derived paths run.py / filter_questions.py compute
internally (so the shell script never has to duplicate that naming logic
by hand and risk drifting out of sync with it), and extracts the "both
high entropy" question_indices for a given threshold from a scan-mode
output file. Keeping this in Python rather than hand-rolled bash/jq
avoids JSON-parsing and quoting bugs in the submission script.

Usage (each subcommand prints exactly one line to stdout, nothing else —
safe to capture directly into a shell variable via $(...)):

  python pipeline_paths.py scan_file <output_dir> <dataset>
      -> {output_dir}/scan_{dataset}_{pair_name}.json
         (matches filter_questions.py's run_scan() naming exactly)

  python pipeline_paths.py distilled_path <dataset>
      -> /scratch-ssd/ms25yt/models/{dataset}_{student_short}_student
         (matches run.py's apply_overrides() naming exactly)

  python pipeline_paths.py question_indices <scan_file> <threshold>
      -> space-separated question indices where BOTH teacher and base
         student have semantic_entropy >= threshold, read from
         <scan_file>'s "thresholds" list. Exits non-zero with a clear
         message on stderr if that exact threshold wasn't scanned.
"""

import json
import sys

from config import cfg
from model_utils import pair_name, short_model_name


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_paths.py "
              "{scan_file|distilled_path|question_indices} ...", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "scan_file":
        output_dir, dataset = sys.argv[2], sys.argv[3]
        pn = pair_name(cfg.teacher_model_name, cfg.student_model_name)
        print(f"{output_dir}/scan_{dataset}_{pn}.json")

    elif cmd == "distilled_path":
        dataset = sys.argv[2]
        student_short = short_model_name(cfg.student_model_name)
        print(f"/scratch-ssd/ms25yt/models/{dataset}_{student_short}_student")

    elif cmd == "question_indices":
        scan_file, threshold_str = sys.argv[2], sys.argv[3]
        threshold = float(threshold_str)
        with open(scan_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        match = [r for r in data["thresholds"] if r["threshold"] == threshold]
        if not match:
            available = [r["threshold"] for r in data["thresholds"]]
            print(f"ERROR: threshold {threshold} not found in {scan_file!r}. "
                  f"Available thresholds: {available}", file=sys.stderr)
            sys.exit(1)
        indices = match[0]["question_indices"]
        if not indices:
            print(f"ERROR: threshold {threshold} matched zero questions "
                  f"(both teacher and base student high-entropy) in "
                  f"{scan_file!r}. Try a lower threshold.", file=sys.stderr)
            sys.exit(1)
        print(" ".join(str(i) for i in indices))

    else:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()