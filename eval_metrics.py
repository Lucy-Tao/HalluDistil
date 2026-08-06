"""
eval_metrics.py -- Phase 6: evaluation metrics on judged files.

Metrics follow Farquhar et al. (2024); computation functions vendored
unmodified from jlko/long_hallucinations eval_utils.py. Bootstrap CIs
are 90% confidence intervals (eval_utils default).

Usage:
  python eval_metrics.py --files judged_data/judged_simpleqa_Qwen3-14B_strict.jsonl ... --plot metrics_round2.png
"""
import argparse, json
import numpy as np

from eval_utils import (
    auroc,
    accuracy_at_quantile,
    area_under_thresholded_accuracy,
    compatible_bootstrap,
)


def load(path, exclude_not_attempted=True):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if exclude_not_attempted:
        recs = [r for r in recs if r["grade"] != "NOT_ATTEMPTED"]
    accuracies = np.array([1.0 if r["grade"] == "CORRECT" else 0.0 for r in recs])
    entropies = np.array([r["semantic_entropy"] for r in recs])
    return accuracies, entropies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--plot", default=None, help="If set, save a 4-panel figure to this path, e.g. metrics.png")
    ap.add_argument("--keep-not-attempted", action="store_true",
                    help="keep NOT_ATTEMPTED rows (default: exclude them before AUROC)")
    args = ap.parse_args()

    results = []

    for path in args.files:
        accuracies, entropies = load(path, exclude_not_attempted=not args.keep_not_attempted)
        y_error = 1 - accuracies          # positive class = incorrect answer
        rng = np.random.default_rng(42)   # fresh rng per file for reproducibility

        a = auroc(y_error, entropies)
        ci = compatible_bootstrap(auroc, rng)(y_error, entropies)
        aurac = area_under_thresholded_accuracy(accuracies, entropies)

        print(f"\n===== {path} =====")
        print(f"  n={len(accuracies)}  accuracy={accuracies.mean():.4f}  "
              f"(n_correct={int(accuracies.sum())})")
        print(f"  AUROC = {a:.4f}  [90% CI: {ci['low']:.4f}-{ci['high']:.4f}, "
              f"SE={ci['std_err']:.4f}]")
        print(f"  AURAC = {aurac:.4f}")
        for q in (0.5, 0.8, 0.9):
            v = accuracy_at_quantile(accuracies, entropies, q)
            print(f"  accuracy@answering-{round(q*100)}% = {v:.4f}  "
                  f"(reject most-uncertain {round((1-q)*100)}%)")

        results.append({"label": path.split("/")[-1].replace("judged_simpleqa_", "").replace(".jsonl", ""),
                        "accuracies": accuracies, "entropies": entropies,
                        "auroc": a, "ci_low": ci["low"], "ci_high": ci["high"], "aurac": aurac})

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(results))
        labels = [r["label"] for r in results]
        prefix = args.plot.rsplit(".", 1)[0]   # "metrics_round2.png" -> "metrics_round2"

        # 1. AUROC with 90% CI
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x, [r["auroc"] for r in results],
                    yerr=[[r["auroc"] - r["ci_low"] for r in results],
                          [r["ci_high"] - r["auroc"] for r in results]],
                    fmt="o", capsize=4)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUROC"); ax.set_title("AUROC (90% bootstrap CI)")
        fig.tight_layout(); fig.savefig(f"{prefix}_auroc.png", dpi=150); plt.close(fig)

        # 2. entropy: correct vs wrong
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, r in enumerate(results):
            ec = r["entropies"][r["accuracies"] == 1]
            ew = r["entropies"][r["accuracies"] == 0]
            ax.scatter(np.full(len(ec), i - 0.15), ec, s=6, alpha=0.5, color="tab:green")
            ax.scatter(np.full(len(ew), i + 0.15), ew, s=6, alpha=0.12, color="tab:red")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("semantic entropy"); ax.set_title("Entropy: correct (green) vs wrong (red)")
        fig.tight_layout(); fig.savefig(f"{prefix}_entropy.png", dpi=150); plt.close(fig)

        # 3. rejection-accuracy curves
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            order = np.argsort(r["entropies"])
            accs = r["accuracies"][order]
            fracs = np.linspace(0.1, 1.0, 50)
            ax.plot(fracs, [accs[:max(1, int(f * len(accs)))].mean() for f in fracs],
                    label=r["label"], lw=1.2)
        ax.set_xlabel("fraction answered (most-certain first)"); ax.set_ylabel("accuracy")
        ax.set_title("Rejection-accuracy curves"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{prefix}_rejection.png", dpi=150); plt.close(fig)

        # 4. accuracy + AURAC bars
        fig, ax = plt.subplots(figsize=(8, 5))
        w = 0.35
        ax.bar(x - w/2, [r["accuracies"].mean() for r in results], w, label="accuracy")
        ax.bar(x + w/2, [r["aurac"] for r in results], w, label="AURAC")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title("Accuracy and AURAC"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{prefix}_acc_aurac.png", dpi=150); plt.close(fig)

        print(f"\nFigures saved: {prefix}_auroc.png, _entropy.png, _rejection.png, _acc_aurac.png")


if __name__ == "__main__":
    main()