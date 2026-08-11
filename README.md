# HalluDistil

Code for an MSc dissertation on whether sequence-level knowledge
distillation transmits a teacher's hallucinations while destroying the
uncertainty signal that would otherwise detect them.

The central measurement is the AUROC of semantic entropy against
incorrectness. A fall toward 0.5 after distillation means entropy no
longer separates right answers from wrong ones, so the student's errors
are harder to catch than the teacher's were.

## Pipeline

Four stages, deliberately decoupled so that each can be rerun alone.

1. **Generate.** One response at T=0.1 for grading, ten at T=1.0 for
   entropy. `generate_responses.py`
2. **Judge.** DeBERTa clusters the ten samples into semantic entropy;
   the official SimpleQA grader labels the low-temperature answer
   CORRECT, INCORRECT or NOT_ATTEMPTED. `judge_responses.py`
3. **Distil.** Hard-label cross entropy on the teacher's
   low-temperature responses, prompt masked. `distill.py` via `run.py`
4. **Evaluate.** AUROC, AURAC and bootstrap intervals.
   `eval_metrics.py`, vendored unmodified from Farquhar et al. (2024)

Stages 1, 2 and 4 are identical for teacher, base student and every
distilled checkpoint, so any difference is attributable to stage 3.

## Running an experiment

Each experiment family has one submission script taking a prompt style
and an epoch count.
sbatch distill_and_eval_v3.sh strict ep20 20 # main line
sbatch distill_and_eval_noskip.sh strict ep20 20 # keep teacher non-answers
sbatch distill_and_eval_se.sh strict loose filter 20
sbatch distill_and_eval_raw.sh strict 5 20
sbatch distill_and_eval_qwen25.sh strict ep10 10
sbatch distill_and_eval_olmo.sh strict ep10 10
**Judging is a separate job.** All of these default to `RUN_JUDGE=0`
and stop after generation. Distillation and generation need about 12GB
while judging needs about 66GB for the grader plus DeBERTa, and SLURM
accounts for GPUs without partitioning device memory, so an all-in-one
job either hangs during model load or dies partway when it shares a
card. Run judging afterwards.
sbatch judge_se.sh <gen_file> <filter|replace|output_dir_name>
`judge_se.sh` checks free memory before touching the device and exits
in seconds rather than minutes when a card is already occupied.

## Interventions

Three ways of changing the distillation targets, each independent and
mutually exclusive with the others.

**Multi-sample targets.** `--raw_samples k` takes k samples at T=1.0
instead of the single greedy target, emitting each as its own training
pair. The student fits the teacher's output distribution rather than
its mode.

**Entropy filter.** `--se_mode filter --se_threshold t` drops items
whose teacher semantic entropy exceeds t. Thresholds sit at midpoints
between adjacent realisable entropy values, since the estimator is
discrete and a cut placed on a realisable value would be decided by
floating-point noise. See `analyse_teacher_entropy.py` for the cut
table.

**Entropy replace.** `--se_mode replace` keeps the item but swaps the
target for an abstention string. Training set size is unchanged, which
removes the data-quantity confound the filter arm has.

## Files

**Core.** `run.py` dispatches by `--mode`. `distill.py` builds the
target set and runs the fine-tune, and is where all three interventions
live. `semantic_utils.py` holds sampling, entailment clustering and the
entropy computation. `generate_responses.py` and `judge_responses.py`
are the two evaluation stages. `config.py` carries shared
hyperparameters, `model_utils.py` handles loading, `data_utils.py`
handles datasets and prompts.

**Long-form.** `generate_longform_responses.py` writes biographies,
`run_factscore_eval.py` decomposes them into atomic claims and judges
each, `compute_claim_entropy.py` computes per-claim entropy on the same
atoms so that entropy and correctness are aligned.
`abstention_rule.py` is the single source of truth for detecting
refusals and is imported rather than reimplemented.

**Analysis.** `eval_metrics.py` for AUROC and AURAC,
`entropy_summary.py` for mean entropy and the fraction of questions
with a single semantic cluster, `analyse_teacher_entropy.py` for the
teacher entropy distribution and threshold table, `rates.py` for
accuracy and abstention, `visualize.py` for per-question cluster plots.

**Subset definition.** `sample_question_indices.py`,
`filter_to_subset.py`, `select_threshold.py`, `build_subset_db.py` and
`sample_and_filter_100_entities.py` are one-off tools that produced the
evaluation subsets. They no longer run, but they define what the
experiments were carried out on, so they are kept.

## Conventions

**The subset is fixed.** `subset_500_seed44_question_indices.json`
defines the 500 short-form questions and every run reads it, so results
are comparable across models and checkpoints.

**Judged files have 500 lines when complete.** Anything shorter is a
partial run. This is the most reliable completion check, more so than
the logs, since a retry loop can report success after an earlier
failure.

**Two append-only logs track provenance.**
`logs/experiment_manifest.log` records pipeline stages with node names,
and `logs/hyperparameter_log.jsonl` records checkpoint identity. The
submission scripts refuse to reuse a checkpoint whose logged
hyperparameters differ from the current request.

**Sampling parameters follow each model family's published
recommendation** rather than one global setting, dispatched by
`sampling_params_for()` in `semantic_utils.py`. Qwen uses 0.8 and 20
for non-thinking mode, Llama 3.1 uses 0.9 with no top_k, and OLMo 2
publishes nothing so falls back to library defaults. AUROC is a rank
statistic, so entropy scales need only agree within a family, which is
where teacher and student are compared.

**Cross-family runs write to their own directories**, suffixed
`_qwen25` and `_olmo`, so that batch operations over the main line do
not pick them up.

## Cluster notes

Oxford oatcloud, `msc` partition, conda environment `haldist`.

`/scratch-ssd/` is node-local. A checkpoint written on one node is
invisible from another, so a job needing an existing checkpoint must be
pinned to the node that holds it. Model caches are node-local for the
same reason and not every node has every model.

**Verify weight completeness with `ls -la snapshots/*/`**, not with a
file count and not by loading the config. A directory can hold the
JSON and tokenizer while the actual blobs are absent, in which case
`AutoConfig.from_pretrained` still succeeds and reports the right
number of layers.

`sbatch` copies the shell script at submission time, so editing a `.sh`
file does not affect queued jobs. Python files are read at execution
time and edits do take effect on jobs that have not yet started. For
job arrays each task reads the script when it starts, so a mid-flight
edit can leave tasks from one array behaving differently.

`sbatch --wrap` runs under `/bin/sh`, which has no `source` builtin, so
a wrapped command that sources `.bashrc` or activates conda fails
silently. Write it as `bash -c` or use a script file.

**Submit in job arrays with a concurrency limit.** Several batches have
been lost to the same pattern, where one job fails in seconds, frees
its slot, and the next queued job lands on the same occupied card, so a
single bad card or one large job can take down twenty in sequence.
`--array=1-N%1` prevents this.
