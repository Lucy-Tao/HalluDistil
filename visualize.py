"""
visualize.py — Token probability distribution plots for Stage 1.

TruthfulQA mode  (Phase 1)
--------------------------
For each prompt, we run one forward pass per model and extract only the
probabilities assigned to the answer-choice tokens (A, B, C, D, ...).
The four panels show these choice probabilities for:
  Panel 1: Teacher
  Panel 2: Teacher  (same distribution; highlights the REAL teacher
                     response used as distillation training data, looked
                     up via distill.load_teacher_distill_response(). Falls
                     back to the argmax/predicted label if no saved sample
                     is found for this question.)
  Panel 3: Base student  (before distillation)
  Panel 4: Distilled student  (after distillation)

Re-normalisation:
  Raw choice probabilities (e.g. P(A) = 0.04) are small because the full
  vocabulary has ~150k tokens.  We re-normalise over the choice tokens so
  the bars sum to 1.0, making visual comparison between panels intuitive.
  Raw values are still printed in the console summary.

Loading models:
  All three models are loaded on demand when run_visualization() is called.
  Loading takes 3-5 minutes; the forward passes themselves take < 1 second.
  For repeated visualization of different question_idx values during an
  interactive session, keep the models in memory between calls.

Usage
-----
  python run.py --mode visualize --question_idx 0
  python run.py --mode visualize --question_idx 1 --scan_file figures/scan_simpleqa_14Bto4B.json
"""

from __future__ import annotations

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import cfg
from data_utils import load_dataset_items
from distill import load_teacher_distill_response
from model_utils import load_model_and_tokenizer, pair_name, short_model_name
from prob_utils import compute_entropy, get_choice_probs, match_response_to_label
from semantic_utils import get_semantic_entropy, load_local_llm_judge, load_nli_model

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "teacher":   "#4C72B0",   # blue
    "highlight": "#55A868",   # green — predicted / sampled label
    "base":      "#C44E52",   # red
    "distilled": "#8172B2",   # purple
}


# ══════════════════════════════════════════════════════════════
# Low-level bar chart helper
# ══════════════════════════════════════════════════════════════

def _plot_choice_bars(
    ax,
    choice_probs: dict,
    title: str,
    color: str,
    highlight_label: str | None = None,
    entropy: float | None = None,
    highlight_label_text: str = "Predicted",
):
    """
    Draw a horizontal bar chart of answer-choice probabilities.

    Args:
        ax             : matplotlib Axes to draw on.
        choice_probs   : dict mapping label -> probability, e.g. {"A": 0.6, "B": 0.3}
                         Values should already be re-normalised (sum to 1.0).
        title          : panel title string.
        color          : bar colour for non-highlighted bars.
        highlight_label: if set, this label's bar is coloured green.
        entropy        : if set, appended to the title as "H = X.XXX nats".
                         Note: entropy is computed over the FULL vocabulary, not
                         just the choice tokens, so it reflects true model uncertainty.
        highlight_label_text: prefix used in the legend, e.g. "Predicted" or
                         "Distillation sample". Lets callers describe WHY this
                         label is highlighted (argmax vs. the real training
                         sample) instead of always saying "Predicted" — the
                         bar colour alone doesn't say which.
    """
    labels = list(choice_probs.keys())
    probs  = np.array([choice_probs[l] for l in labels])

    # Reverse so highest-probability option appears at the top
    labels = labels[::-1]
    probs  = probs[::-1]

    bar_colors = [
        PALETTE["highlight"] if l == highlight_label else color
        for l in labels
    ]

    bars = ax.barh(labels, probs, color=bar_colors, edgecolor="white", linewidth=0.8)

    ax.set_xlabel("Probability (re-normalised over choices)", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bars with probability values
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}",
            va="center", fontsize=9, color="#333333",
        )

    # Build title with optional entropy
    full_title = title
    if entropy is not None:
        full_title += f"\n(choice H = {entropy:.3f} nats)"
    ax.set_title(full_title, fontsize=11, fontweight="bold", pad=8)

    if highlight_label:
        patch = mpatches.Patch(
            color=PALETTE["highlight"],
            label=f"{highlight_label_text}: '{highlight_label}'"
        )
        ax.legend(handles=[patch], fontsize=9, loc="lower right")


# ══════════════════════════════════════════════════════════════
# Single-prompt 4-panel visualization
# ══════════════════════════════════════════════════════════════

def run_visualization(question_idx: int = 0, scan_file: str | None = None):
    """
    Public entry point called by run.py.  Dispatches to the correct
    visualization path based on cfg.dataset:

      "truthfulqa" / "gpqa" / "mmlu_pro"  ->  _run_mcq_visualization()
          Single forward pass, choice-token probabilities, deterministic.

      "simpleqa"                          ->  _run_simpleqa_visualization()
          Multiple sampled responses per model, NLI-based semantic
          clustering, frequency distribution over meaning-clusters.

    Args:
        scan_file: optional path to a filter_questions.py scan output JSON.
            For SimpleQA: if provided, teacher and base student responses
            for question_idx are read straight from this file instead of
            loading those models and re-generating — only the distilled
            student is loaded and sampled fresh. Saves ~40GB of VRAM and
            several minutes per question when you've already run a scan.
    """
    if cfg.dataset == "simpleqa":
        _run_simpleqa_visualization(question_idx, scan_file=scan_file)
    else:
        _run_mcq_visualization(question_idx)


def _run_mcq_visualization(question_idx: int = 0):
    """
    MCQ visualization path — used for truthfulqa / gpqa / mmlu_pro.

    Load all three models, run one forward pass each, and plot the four panels.

    Panel descriptions
    ------------------
    Panel 1 — Teacher full choice distribution
        Shows how uncertain the teacher is across answer options.
        A flat distribution means high uncertainty; a peaked distribution means
        the teacher is confident (possibly overconfident / hallucinating).

    Panel 2 — Teacher distribution with the REAL distillation sample highlighted
        Identical bar heights to Panel 1, but highlights whichever label the
        teacher's response ACTUALLY used as distillation training data
        corresponds to (looked up via distill.load_teacher_distill_response()
        and matched back to a label via prob_utils.match_response_to_label()).
        This can differ from the argmax label whenever the teacher's
        distribution isn't sharply peaked, since the training response was
        sampled, not argmax-decoded. Falls back to highlighting the argmax
        label if no saved sample is found for this question.

    Panel 3 — Base student choice distribution (before distillation)
        The student's natural distribution before seeing any teacher responses.
        May differ substantially from the teacher if the student has lower capacity.

    Panel 4 — Distilled student choice distribution (after distillation)
        Key panel: has the distribution become more peaked than Panel 1?
        If yes, distillation compressed uncertainty toward the teacher's prediction
        — the core hallucination-confidence phenomenon your Stage 1 study examines.
    """
    print("\n" + "=" * 60)
    print(f"VISUALIZATION  (TruthfulQA, question_idx={question_idx})")
    print("=" * 60)

    # ── Load dataset item ─────────────────────────────────────
    items = load_dataset_items(cfg.dataset, num_samples=question_idx + 1)
    item  = items[question_idx]
    prompt  = item["prompt"]
    choices = item["choices"]   # e.g. ["A", "B", "C", "D"]

    print(f"\nQuestion [{question_idx}]: {item['question']}")
    print(f"Choices: {choices}")
    print(f"\nFull prompt:\n{prompt}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Teacher ───────────────────────────────────────────────
    print("[1/3] Loading teacher model...")
    teacher_model, teacher_tok = load_model_and_tokenizer(
        cfg.teacher_model_name, device_map=cfg.device_map
    )
    teacher_result = get_choice_probs(teacher_model, teacher_tok, prompt, choices)
    del teacher_model
    torch.cuda.empty_cache()
    print(f"  Teacher done. Predicted: {teacher_result['predicted_label']}")

    # ── Real distillation sample lookup (for Panel 2) ──────────
    # Panel 2 should highlight the label the teacher's response ACTUALLY
    # used for distillation training landed on — not just the argmax label
    # (those can differ whenever the teacher's distribution isn't sharply
    # peaked, since the training response was sampled, not argmax-decoded).
    distill_response = load_teacher_distill_response(
        cfg.dataset, cfg.teacher_model_name, question_idx
    )
    distill_label = None
    if distill_response is not None:
        distill_label = match_response_to_label(distill_response, choices)
        if distill_label is None:
            print(f"  WARNING: saved distillation response {distill_response!r} "
                  f"could not be matched to any choice label {choices}. "
                  f"Falling back to predicted_label for Panel 2 highlight.")

    # ── Base student ──────────────────────────────────────────
    print("[2/3] Loading base student model...")
    base_model, base_tok = load_model_and_tokenizer(
        cfg.student_model_name, device_map=cfg.device_map
    )
    base_result = get_choice_probs(base_model, base_tok, prompt, choices)
    del base_model
    torch.cuda.empty_cache()
    print(f"  Base student done. Predicted: {base_result['predicted_label']}")

    # ── Distilled student ─────────────────────────────────────
    print("[3/3] Loading distilled student model...")
    if not os.path.exists(cfg.distilled_model_path):
        raise FileNotFoundError(
            f"Distilled model not found at '{cfg.distilled_model_path}'.\n"
            "Run  python run.py --mode distill  first."
        )
    dist_model, dist_tok = load_model_and_tokenizer(
        cfg.distilled_model_path, device_map=cfg.device_map
    )
    dist_result = get_choice_probs(dist_model, dist_tok, prompt, choices)
    del dist_model
    torch.cuda.empty_cache()
    print(f"  Distilled student done. Predicted: {dist_result['predicted_label']}")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    question_preview = item["question"][:100]
    fig.suptitle(
        f"Answer-Choice Probability Distributions  |  Q{question_idx}\n"
        f'"{question_preview}{"..." if len(item["question"]) > 100 else ""}"\n'
        f"Teacher: {short_model_name(cfg.teacher_model_name)}  |  "
        f"Base/Distilled Student: {short_model_name(cfg.student_model_name)}  |  "
        f"Correct answer: {item['answer']}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    # Panel 1: Teacher full distribution
    _plot_choice_bars(
        axes[0, 0],
        teacher_result["choice_probs"],
        title="1  Teacher — Choice Distribution",
        color=PALETTE["teacher"],
        entropy=teacher_result["choice_entropy"],
    )

    # Panel 2: Teacher with the REAL distillation sample highlighted
    # (falls back to predicted_label if no saved sample was found/matched)
    if distill_label is not None:
        panel2_highlight = distill_label
        panel2_title = "2  Teacher — Distillation Sample Highlighted"
        panel2_legend_text = "Distillation sample"
    else:
        panel2_highlight = teacher_result["predicted_label"]
        panel2_title = "2  Teacher — Predicted Label Highlighted (no saved sample)"
        panel2_legend_text = "Predicted"
    _plot_choice_bars(
        axes[0, 1],
        teacher_result["choice_probs"],
        title=panel2_title,
        color=PALETTE["teacher"],
        highlight_label=panel2_highlight,
        highlight_label_text=panel2_legend_text,
    )

    # Panel 3: Base student
    _plot_choice_bars(
        axes[1, 0],
        base_result["choice_probs"],
        title="3  Base Student — Before Distillation",
        color=PALETTE["base"],
        highlight_label=base_result["predicted_label"],
        entropy=base_result["choice_entropy"],
    )

    # Panel 4: Distilled student
    _plot_choice_bars(
        axes[1, 1],
        dist_result["choice_probs"],
        title="4  Distilled Student — After Distillation",
        color=PALETTE["distilled"],
        highlight_label=dist_result["predicted_label"],
        entropy=dist_result["choice_entropy"],
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(
        cfg.output_dir,
        f"choice_dist_{cfg.dataset}_{pair_name(cfg.teacher_model_name, cfg.student_model_name)}"
        f"_q{question_idx:03d}.png",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved -> {save_path}")

    # ── Console summary ───────────────────────────────────────
    print("\n" + "-" * 55)
    print(f"{'Label':<8} {'Teacher':>10} {'Base':>10} {'Distilled':>12}")
    print("-" * 55)
    for label in choices:
        t = teacher_result["choice_probs"][label]
        b = base_result["choice_probs"][label]
        d = dist_result["choice_probs"][label]
        print(f"  {label:<6} {t:>10.4f} {b:>10.4f} {d:>12.4f}")
    print("-" * 55)
    print(f"  Predicted  "
          f"{'[' + teacher_result['predicted_label'] + ']':>10} "
          f"{'[' + base_result['predicted_label'] + ']':>10} "
          f"{'[' + dist_result['predicted_label'] + ']':>12}")
    if distill_label is not None:
        print(f"  Teacher's actual distillation sample -> label '{distill_label}' "
              f"(response: {distill_response!r})")
    else:
        print(f"  No matched distillation sample for teacher on this question "
              f"(Panel 2 fell back to predicted_label).")
    n_choices = len(choices)
    max_h = float(__import__("math").log(n_choices))
    print(f"\n  Choice entropy (nats, max={max_h:.3f} for {n_choices} options):")
    print(f"    Teacher:           {teacher_result['choice_entropy']:.4f}")
    print(f"    Base student:      {base_result['choice_entropy']:.4f}")
    print(f"    Distilled student: {dist_result['choice_entropy']:.4f}")
    gap  = dist_result["choice_entropy"] - teacher_result["choice_entropy"]
    sign = "more confident" if gap < 0 else "less confident"
    print(f"    Distilled vs Teacher: {gap:+.4f}  ({sign})")

    # ── Save JSON ─────────────────────────────────────────────
    results = {
        "question_idx":   question_idx,
        "question":       item["question"],
        "choices":        choices,
        "teacher":  {
            "choice_probs":    teacher_result["choice_probs"],
            "raw_probs":       teacher_result["raw_choice_probs"],
            "predicted":       teacher_result["predicted_label"],
            "choice_entropy":     teacher_result["choice_entropy"],
            "distill_sample_response": distill_response,
            "distill_sample_label":    distill_label,
        },
        "base": {
            "choice_probs":    base_result["choice_probs"],
            "raw_probs":       base_result["raw_choice_probs"],
            "predicted":       base_result["predicted_label"],
            "choice_entropy":     base_result["choice_entropy"],
        },
        "distilled": {
            "choice_probs":    dist_result["choice_probs"],
            "raw_probs":       dist_result["raw_choice_probs"],
            "predicted":       dist_result["predicted_label"],
            "choice_entropy":     dist_result["choice_entropy"],
        },
        "choice_entropy_gap_distilled_vs_teacher": gap,
    }
    json_path = os.path.join(
        cfg.output_dir,
        f"results_{cfg.dataset}_{pair_name(cfg.teacher_model_name, cfg.student_model_name)}"
        f"_q{question_idx:03d}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON -> {json_path}")
    return results

# ══════════════════════════════════════════════════════════════
# SimpleQA visualization (Phase 2 — semantic entropy)
# ══════════════════════════════════════════════════════════════

def _plot_semantic_bars(
    ax,
    cluster_probs: dict,
    title: str,
    color: str,
    highlight_response: str | None = None,
    entropy: float | None = None,
    max_label_len: int = 40,
    highlight_text: str = "Most frequent answer",
):
    """
    Draw a horizontal bar chart of semantic-cluster frequencies.

    Unlike _plot_choice_bars (fixed A/B/C/D labels), the bar labels here
    are full response strings, which can be long — they are truncated to
    max_label_len characters for display.

    IMPORTANT: bars are placed at integer y-positions (0, 1, 2, ...), NOT
    at the (truncated) label strings themselves. Passing the truncated
    strings directly to ax.barh() as the y-values is a real bug we hit in
    practice: matplotlib treats string y-values as categorical, and
    DEDUPLICATES identical strings into a single row — so if two
    DIFFERENT clusters happen to share the same first max_label_len
    characters (very likely when a model answers in full, rambling
    sentences that all start the same way — e.g. a base/un-distilled
    student that isn't following a "short answer only" instruction as
    reliably as the teacher), their bars get drawn ON TOP of each other
    at the same position. The chart then shows what looks like a single
    bar even though semantic_entropy (computed from the real, untruncated
    cluster_probs) is correctly nonzero — the data was always right, only
    the rendering was silently dropping rows. Using numeric positions for
    placement and a separate set_yticklabels() call for the (possibly
    duplicate-looking) display text avoids this entirely: every cluster
    always gets its own row, no matter what its truncated label looks like.

    Args:
        cluster_probs : dict mapping representative response string -> frequency.
        highlight_response: if set, that bar is coloured green.
        entropy       : semantic entropy value, appended to the title.
        highlight_text: legend caption describing WHY this bar is highlighted,
                        e.g. "Most frequent answer" or "Distillation sample".
                        Lets callers distinguish "this is the mode" from
                        "this is the actual response used to train the
                        student" — the bar colour alone doesn't say which.
    """
    # Sort clusters by frequency descending so the most common answer is on top
    sorted_items = sorted(cluster_probs.items(), key=lambda x: x[1], reverse=True)
    raw_texts = [text for text, _ in sorted_items]
    probs     = np.array([prob for _, prob in sorted_items])

    labels = [text[:max_label_len] + ("..." if len(text) > max_label_len else "")
              for text in raw_texts]
    # If truncation makes two or more DIFFERENT clusters look identical,
    # append a disambiguating suffix so the chart stays readable (the
    # underlying bar PLACEMENT no longer depends on this — see note above
    # — but identical-looking labels on different rows is still confusing
    # to read, so fix that too rather than just the data-loss bug).
    seen_counts: dict[str, int] = {}
    disambiguated = []
    for lbl in labels:
        seen_counts[lbl] = seen_counts.get(lbl, 0) + 1
        disambiguated.append(lbl)
    label_occurrence = {}
    for i, lbl in enumerate(labels):
        if seen_counts[lbl] > 1:
            label_occurrence[lbl] = label_occurrence.get(lbl, 0) + 1
            disambiguated[i] = f"{lbl} [{label_occurrence[lbl]}/{seen_counts[lbl]}]"
    labels = disambiguated

    # Reverse so the highest-frequency cluster appears at the top of the chart
    labels    = labels[::-1]
    probs     = probs[::-1]
    raw_texts = raw_texts[::-1]

    y_pos = np.arange(len(labels))  # numeric placement — see docstring above

    bar_colors = [
        PALETTE["highlight"] if raw == highlight_response else color
        for raw in raw_texts
    ]

    bars = ax.barh(y_pos, probs, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Frequency among sampled responses", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.2f}",
            va="center", fontsize=9, color="#333333",
        )

    full_title = title
    if entropy is not None:
        full_title += f"\n(semantic H = {entropy:.3f} nats)"
    ax.set_title(full_title, fontsize=11, fontweight="bold", pad=8)

    if highlight_response:
        patch = mpatches.Patch(
            color=PALETTE["highlight"],
            label=highlight_text
        )
        ax.legend(handles=[patch], fontsize=9, loc="lower right")


def _run_simpleqa_visualization(question_idx: int = 0,
                                scan_file: str | None = None):
    """
    SimpleQA visualization path — semantic entropy over sampled responses.

    For each model (teacher, base student, distilled student):
      1. Sample cfg.num_semantic_samples complete responses to the prompt.
      2. Cluster them by bidirectional NLI entailment.
      3. Compute the frequency distribution over semantic clusters and its
         entropy.

    Panel descriptions
    ------------------
    Panel 1 — Teacher semantic-cluster distribution
        Shows how consistent the teacher's answers are across samples.
        Multiple large clusters = teacher itself is uncertain/inconsistent.

    Panel 2 — Teacher distribution with the REAL distillation sample highlighted
        Same bars as Panel 1, but instead of highlighting whichever cluster
        happens to be most frequent, this highlights the cluster containing
        the teacher's response that was ACTUALLY used as distillation
        training data for this question (looked up via
        distill.load_teacher_distill_response()). That exact response is
        reused as one of the N samples in Panel 1 too — so only N-1 new
        samples are drawn, saving one generation call and guaranteeing the
        "what the student was trained on" point is represented in the
        clustering rather than possibly missed by chance.
        Falls back to highlighting the modal answer (old behaviour) if no
        saved distillation sample is found for this question.

    Panel 3 — Base student semantic-cluster distribution (before distillation)

    Panel 4 — Distilled student semantic-cluster distribution (after distillation)
        Key panel: has the distribution become more peaked (lower entropy)
        than Panel 1? If yes, distillation compressed answer diversity toward
        a single (possibly hallucinated) answer — the Stage 1 hypothesis,
        now tested in the open-ended generation setting.
    """
    print("\n" + "=" * 60)
    print(f"VISUALIZATION  (SimpleQA, question_idx={question_idx}, "
          f"N={cfg.num_semantic_samples} samples/model)")
    print("=" * 60)

    # ── Load dataset item ─────────────────────────────────────
    items  = load_dataset_items(cfg.dataset, num_samples=question_idx + 1)
    item   = items[question_idx]
    prompt = item["prompt"]

    print(f"\nQuestion [{question_idx}]: {item['question']}")
    print(f"\nFull prompt:\n{prompt}\n")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Entailment judge: loaded once, shared across all three generation
    # models. Backend selected via cfg.entailment_backend ("deberta" or
    # "llm") — see semantic_utils.py module docstring for details.
    print(f"[Judge] Loading entailment judge (backend={cfg.entailment_backend!r})...")
    if cfg.entailment_backend == "llm":
        nli_model, nli_tokenizer = load_local_llm_judge(cfg.entailment_llm_model_name)
    elif cfg.entailment_backend == "deberta":
        nli_model, nli_tokenizer = load_nli_model(cfg.nli_model_name)
    else:
        raise ValueError(
            f"Unknown cfg.entailment_backend={cfg.entailment_backend!r}. "
            f"Use 'deberta' or 'llm'."
        )

    def _run_one_model(model_name: str, label: str, fixed_response: str | None = None) -> dict:
        m, tok = load_model_and_tokenizer(model_name, device_map=cfg.device_map)
        result = get_semantic_entropy(
            m, tok, prompt,
            nli_model=nli_model, nli_tokenizer=nli_tokenizer,
            n_samples=cfg.num_semantic_samples,
            temperature=cfg.semantic_sample_temperature,
            max_new_tokens=cfg.semantic_max_new_tokens,
            threshold=cfg.entailment_threshold,
            fixed_response=fixed_response,
            judge_backend=cfg.entailment_backend,
            question=item["question"],
        )
        del m
        torch.cuda.empty_cache()
        print(f"  {label} done. Sampled responses: {result['raw_responses']}")
        print(f"  {label} predicted: '{result['predicted_response']}'  "
              f"(semantic entropy = {result['semantic_entropy']:.4f})")
        return result

    def _reuse_scan_record(record: dict, label: str) -> dict:
        """
        Re-cluster an already-sampled record from a scan file with the
        currently loaded judge, instead of loading the model and sampling
        fresh responses. The raw_responses (10 strings) are taken straight
        from the record; only the clustering / entropy computation is re-run.
        """
        from semantic_utils import cluster_by_entailment, compute_semantic_distribution
        clusters = cluster_by_entailment(
            record["raw_responses"], nli_model, nli_tokenizer,
            cfg.entailment_threshold,
            backend=cfg.entailment_backend,
            question=item["question"],
        )
        dist = compute_semantic_distribution(record["raw_responses"], clusters)
        result = {
            **dist,
            "raw_responses": record["raw_responses"],
        }
        # fixed_response is the low_temp_response from the scan (the distillation sample)
        fixed = record.get("low_temp_response")
        if fixed is not None:
            result["fixed_response"] = fixed
            result["fixed_response_cluster"] = None
            from collections import Counter
            fixed_indices = [i for i, r in enumerate(record["raw_responses"]) if r == fixed]
            if fixed_indices:
                for cluster in clusters:
                    if fixed_indices[0] in cluster:
                        member_texts = [record["raw_responses"][i] for i in cluster]
                        result["fixed_response_cluster"] = Counter(member_texts).most_common(1)[0][0]
                        break
        print(f"  {label} done (reused scan responses). "
              f"Sampled responses: {result['raw_responses']}")
        print(f"  {label} predicted: '{result['predicted_response']}'  "
              f"(semantic entropy = {result['semantic_entropy']:.4f})")
        return result

    # ── Load scan records if available ────────────────────────
    scan_teacher_record = None
    scan_student_record = None
    if scan_file is not None:
        import json as _json
        with open(scan_file, "r", encoding="utf-8") as _f:
            _scan = _json.load(_f)
        _t_by_idx = {r["question_idx"]: r for r in _scan.get("teacher_records", [])}
        _s_by_idx = {r["question_idx"]: r for r in _scan.get("student_records", [])}
        if question_idx in _t_by_idx:
            scan_teacher_record = _t_by_idx[question_idx]
            print(f"  Found teacher record for Q{question_idx} in scan file "
                  f"({len(scan_teacher_record['raw_responses'])} responses) — "
                  f"teacher model will NOT be loaded.")
        else:
            print(f"  Q{question_idx} not in scan file teacher_records — "
                  f"will load teacher model and sample fresh.")
        if question_idx in _s_by_idx:
            scan_student_record = _s_by_idx[question_idx]
            print(f"  Found base student record for Q{question_idx} in scan file "
                  f"({len(scan_student_record['raw_responses'])} responses) — "
                  f"base student model will NOT be loaded.")
        else:
            print(f"  Q{question_idx} not in scan file student_records — "
                  f"will load base student model and sample fresh.")

    # Look up the teacher's distillation sample (low-temp response) for
    # Panel 2 highlighting. If scan_file has this question, the record's
    # low_temp_response IS the distillation sample (that's what was saved
    # there). Otherwise fall back to load_teacher_distill_response().
    if scan_teacher_record is not None:
        distill_response = scan_teacher_record.get("low_temp_response")
    else:
        distill_response = load_teacher_distill_response(
            cfg.dataset, cfg.teacher_model_name, question_idx
        )
    if distill_response is None:
        print("  No saved teacher distillation sample for this question — "
              "Panel 2 will fall back to highlighting the modal answer.")

    print("\n[1/3] Teacher model...")
    if scan_teacher_record is not None:
        teacher_result = _reuse_scan_record(scan_teacher_record, "Teacher")
        # Ensure fixed_response is set correctly for Panel 2
        if distill_response is not None and "fixed_response_cluster" not in teacher_result:
            teacher_result["fixed_response"] = distill_response
            teacher_result["fixed_response_cluster"] = teacher_result.get("predicted_response")
    else:
        teacher_result = _run_one_model(
            cfg.teacher_model_name, "Teacher", fixed_response=distill_response
        )

    print("\n[2/3] Base student model...")
    if scan_student_record is not None:
        base_result = _reuse_scan_record(scan_student_record, "Base student")
    else:
        base_result = _run_one_model(cfg.student_model_name, "Base student")

    print("\n[3/3] Distilled student model...")
    if not os.path.exists(cfg.distilled_model_path):
        raise FileNotFoundError(
            f"Distilled model not found at '{cfg.distilled_model_path}'.\n"
            "Run  python run.py --mode distill  first."
        )
    dist_result = _run_one_model(cfg.distilled_model_path, "Distilled student")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    question_preview = item["question"][:100]
    fig.suptitle(
        f"Semantic-Cluster Frequency Distributions  |  Q{question_idx}\n"
        f'"{question_preview}{"..." if len(item["question"]) > 100 else ""}"  '
        f"(N={cfg.num_semantic_samples} samples per model)\n"
        f"Teacher: {short_model_name(cfg.teacher_model_name)}  |  "
        f"Base/Distilled Student: {short_model_name(cfg.student_model_name)}  |  "
        f"Gold answer: {item['answer']}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    _plot_semantic_bars(
        axes[0, 0], teacher_result["cluster_probs"],
        title="1  Teacher — Semantic Distribution",
        color=PALETTE["teacher"],
        entropy=teacher_result["semantic_entropy"],
    )
    # Panel 2 highlight: the cluster containing the REAL distillation sample
    # if we found/reused one, else fall back to the modal answer (old
    # behaviour) and say so in the title.
    panel2_highlight = teacher_result.get("fixed_response_cluster")
    if panel2_highlight is not None:
        panel2_title = "2  Teacher — Distillation Sample Highlighted"
        panel2_legend_text = "Distillation sample"
    else:
        panel2_highlight = teacher_result["predicted_response"]
        panel2_title = "2  Teacher — Most Frequent Answer Highlighted (no saved sample)"
        panel2_legend_text = "Most frequent answer"
    _plot_semantic_bars(
        axes[0, 1], teacher_result["cluster_probs"],
        title=panel2_title,
        color=PALETTE["teacher"],
        highlight_response=panel2_highlight,
        highlight_text=panel2_legend_text,
    )
    _plot_semantic_bars(
        axes[1, 0], base_result["cluster_probs"],
        title="3  Base Student — Before Distillation",
        color=PALETTE["base"],
        highlight_response=base_result["predicted_response"],
        entropy=base_result["semantic_entropy"],
    )
    _plot_semantic_bars(
        axes[1, 1], dist_result["cluster_probs"],
        title="4  Distilled Student — After Distillation",
        color=PALETTE["distilled"],
        highlight_response=dist_result["predicted_response"],
        entropy=dist_result["semantic_entropy"],
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(
        cfg.output_dir,
        f"semantic_dist_{cfg.dataset}_{pair_name(cfg.teacher_model_name, cfg.student_model_name)}"
        f"_q{question_idx:03d}.png",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved -> {save_path}")

    # ── Console summary ───────────────────────────────────────
    t_se = teacher_result["semantic_entropy"]
    b_se = base_result["semantic_entropy"]
    d_se = dist_result["semantic_entropy"]

    print("\n" + "-" * 55)
    print("Semantic entropy summary (nats)")
    print("-" * 55)
    print(f"  Teacher:           {t_se:.4f}")
    print(f"  Base student:      {b_se:.4f}")
    print(f"  Distilled student: {d_se:.4f}")
    gap  = d_se - t_se
    sign = "more confident" if gap < 0 else "less confident"
    print(f"  Distilled vs Teacher: {gap:+.4f}  ({sign})")
    if teacher_result.get("fixed_response_cluster") is not None:
        print(f"  Teacher's actual distillation sample: {distill_response!r}  "
              f"-> cluster '{teacher_result['fixed_response_cluster']}'")
    else:
        print("  No matched distillation sample for teacher on this question "
              "(Panel 2 fell back to modal answer).")

    # ── Save JSON ─────────────────────────────────────────────
    results = {
        "question_idx": question_idx,
        "question":     item["question"],
        "num_samples":  cfg.num_semantic_samples,
        "teacher": {
            "raw_responses":      teacher_result["raw_responses"],
            "cluster_probs":      teacher_result["cluster_probs"],
            "predicted_response": teacher_result["predicted_response"],
            "semantic_entropy":   t_se,
            "distill_sample_response": distill_response,
            "distill_sample_cluster":  teacher_result.get("fixed_response_cluster"),
        },
        "base": {
            "raw_responses":      base_result["raw_responses"],
            "cluster_probs":      base_result["cluster_probs"],
            "predicted_response": base_result["predicted_response"],
            "semantic_entropy":   b_se,
        },
        "distilled": {
            "raw_responses":      dist_result["raw_responses"],
            "cluster_probs":      dist_result["cluster_probs"],
            "predicted_response": dist_result["predicted_response"],
            "semantic_entropy":   d_se,
        },
        "semantic_entropy_gap_distilled_vs_teacher": gap,
    }
    json_path = os.path.join(
        cfg.output_dir,
        f"results_{cfg.dataset}_{pair_name(cfg.teacher_model_name, cfg.student_model_name)}"
        f"_q{question_idx:03d}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON -> {json_path}")
    return results



# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_idx", type=int, default=0)
    parser.add_argument("--scan_file", type=str, default=None)
    args = parser.parse_args()
    run_visualization(args.question_idx, scan_file=args.scan_file)