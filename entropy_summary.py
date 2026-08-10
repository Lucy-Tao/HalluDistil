"""Mean semantic entropy and single-cluster fraction per judged file."""
import sys, json, os
import numpy as np

print(f"{'mean_SE':>8} {'sd':>6} {'p_single':>9} {'n':>5}   file")
for f in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f) if l.strip()]
    recs = [r for r in recs if r.get("grade") != "NOT_ATTEMPTED"]
    se = np.array([r["semantic_entropy"] for r in recs])
    single = np.mean(se == 0.0)
    print(f"{se.mean():8.4f} {se.std():6.4f} {single:9.1%} {len(se):5d}   {os.path.basename(f)}")
