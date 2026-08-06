"""
decompose_atomic_facts.py — Phase 2: break each teacher/student bio in
answered_both.jsonl into atomic facts, following FActScore's sentence-level
decomposition protocol (Min et al. 2023, factscore/atomic_facts.py).

WHAT THIS FOLLOWS FROM THE OFFICIAL METHOD, AND WHAT'S DIFFERENT
------------------------------------------------------------------
Followed:
  - Sentence-level decomposition: the response is split into sentences
    FIRST (here via spaCy, same as official), and each sentence is
    decomposed independently, one LLM call per sentence.
  - Prompt format: "Please breakdown the following sentence into
    independent facts: Sentence: ... Independent Facts: - ..." with
    few-shot demonstrations, same shape as the official prompt.
  - Demonstrations: the SENTENCES and base fact style in _DEMOS below are
    the real official demons.json examples (Min et al. 2023) — the
    project owner downloaded the actual data package from the official
    Google Drive link and shared its contents.

Different (necessarily):
  - Model: official used InstructGPT (text-davinci-003), which OpenAI
    has since deprecated — it is no longer callable via any API. This
    uses a local open-weight instruction model instead (default:
    Qwen2.5-32B-Instruct, see cfg.decomposition_model_name). 
  - Pronoun resolution: official demons.json does NOT resolve pronouns in
    its output facts (its actual example "He is also a successful
    producer..." decomposes to "He is successful.", "He is a producer.",
    etc. — "He" stays as-is). That doesn't work for this project: each
    fact needs to be independently checkable (FActScore-style
    verification) AND independently rewritable into a standalone question
    (semantic-entropy-style resampling) later, and "He is a producer" is
    neither once separated from its source sentence. So here the pronoun
    IS resolved to the entity's name in the fact outputs — a deliberate,
    documented deviation the project owner chose explicitly (over the
    alternative of matching official exactly and adding pronoun
    resolution as a separate later step); see _DEMOS below and project
    chat history for specifics.
  - Markdown cleanup (NOT part of official FActScore at all): earlier
    generations came out of a chat model with **bold**, ### headers, and
    "- " bullet lists before the generation prompt was updated to
    explicitly request plain prose — official FActScore's demonstrations
    assume plain prose throughout, matching the updated generation setup.
    A defensive markdown-strip is still applied here as a fallback (see
    clean_markdown()) since an instruction to the generator isn't a 100%
    guarantee, but it should now be a no-op on most responses.

Checkpointed incrementally per question_idx — safe to resume after a
SLURM timeout/crash, same pattern as the rest of this pipeline. Each run
processes ONE model_role (teacher or student); run it twice, once per
role, with two different --output paths (model_role is encoded in the
filename you choose, not written into the records — see --model_role
below).

Requires spaCy's small English pipeline (sentence segmentation only —
no need for the larger model):
    pip install spacy --break-system-packages
    python -m spacy download en_core_web_sm

Usage
-----
  python decompose_atomic_facts.py \\
      --input ~/SimpleQA/gen_longform_data/answered_both.jsonl \\
      --model_role teacher \\
      --output ~/SimpleQA/gen_longform_data/atomic_facts_teacher.jsonl

  python decompose_atomic_facts.py \\
      --input ~/SimpleQA/gen_longform_data/answered_both.jsonl \\
      --model_role student \\
      --output ~/SimpleQA/gen_longform_data/atomic_facts_student.jsonl

Output: one jsonl line per (question_idx, sentence_idx) for the chosen
model_role:
  {
    "question_idx": int,
    "entity": str,
    "sentence_idx": int,
    "sentence": str,           # the cleaned sentence this came from
    "facts": [str, ...],       # atomic facts extracted from that sentence
  }
"""
import argparse
import json
import os
import re

import torch

from config import cfg
from model_utils import load_model_and_tokenizer


# ══════════════════════════════════════════════════════════════
# Step 0 — Markdown cleanup + sentence splitting
# ══════════════════════════════════════════════════════════════

# Lines that are pure formatting, not prose — drop entirely before sentence
# splitting rather than trying to decompose them into "facts".
_HEADER_LINE = re.compile(r"^\s*#{1,6}\s*.*$")
_HR_LINE = re.compile(r"^\s*(-{3,}|\*{3,})\s*$")
_BULLET_PREFIX = re.compile(r"^\s*[-*]\s+")
_NUMBERED_PREFIX = re.compile(r"^\s*\d+[.)]\s+")
_BOLD_ITALIC = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_INLINE_CODE = re.compile(r"`([^`]+)`")


def clean_markdown(text: str) -> str:
    
    lines = []
    for line in text.split("\n"):
        if _HEADER_LINE.match(line) or _HR_LINE.match(line):
            continue
        line = _BULLET_PREFIX.sub("", line)
        line = _NUMBERED_PREFIX.sub("", line)
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = _BOLD_ITALIC.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    return cleaned


_SPACY_NLP = None


def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])
        except OSError as e:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. Install with:\n"
                "  pip install spacy --break-system-packages\n"
                "  python -m spacy download en_core_web_sm"
            ) from e
    return _SPACY_NLP


def split_sentences(text: str, min_words: int = 2) -> list[str]:
    """
    Clean any stray markdown, then split into sentences via a single
    whole-document spaCy pass — same shape as official FActScore's flow.
    Sentences with fewer than min_words words after cleaning are dropped
    (rare with plain-prose input; mainly guards against an empty line or
    leftover fragment slipping through).
    """
    cleaned = clean_markdown(text)
    nlp = _get_spacy_nlp()
    sentences = []
    for sent in nlp(cleaned).sents:
        s = sent.text.strip()
        if not s:
            continue
        if len(s.split()) < min_words:
            continue
        sentences.append(s)
    return sentences


# ══════════════════════════════════════════════════════════════
# Step 1 — Atomic fact decomposition (one LLM call per sentence)
# ══════════════════════════════════════════════════════════════

# Few-shot demonstrations. The SENTENCES and base fact style below are
# taken from the official FActScore demons.json (Min et al. 2023) — the
# project owner downloaded the real data package from the official Google
# Drive link and shared its contents, so these are the actual official
# demonstrations, not invented ones (an earlier version of this script
# had made-up examples because the official file wasn't available yet —
# replaced now that it is).

_DEMOS = [
    {
        "entity": "Michael Collins",
        "sentence": "Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.",
        "facts": [
            "Michael Collins was born on October 31, 1930.",
            "Michael Collins is retired.",
            "Michael Collins is an American.",
            "Michael Collins was an astronaut.",
            "Michael Collins was a test pilot.",
            "Michael Collins was the Command Module Pilot.",
            "Michael Collins was the Command Module Pilot for the Apollo 11 mission.",
            "Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.",
        ],
    },
    {
        "entity": "McCoy",
        "sentence": "During his professional career, McCoy played for the Broncos, the San Diego Chargers, the Minnesota Vikings, and the Jacksonville Jaguars.",
        "facts": [
            "McCoy played for the Broncos.",
            "McCoy played for the Broncos during his professional career.",
            "McCoy played for the San Diego Chargers.",
            "McCoy played for the San Diego Chargers during his professional career.",
            "McCoy played for the Minnesota Vikings.",
            "McCoy played for the Minnesota Vikings during his professional career.",
            "McCoy played for the Jacksonville Jaguars.",
            "McCoy played for the Jacksonville Jaguars during his professional career.",
        ],
    },
    {
        "entity": "Quincy Jones",
        "sentence": "He is also a successful producer and engineer, having worked with a wide variety of artists, including Willie Nelson, Tim McGraw, and Taylor Swift.",
        "facts": [
            "Quincy Jones is successful.",
            "Quincy Jones is a producer.",
            "Quincy Jones is an engineer.",
            "Quincy Jones has worked with a wide variety of artists.",
            "Willie Nelson is an artist.",
            "Quincy Jones has worked with Willie Nelson.",
            "Tim McGraw is an artist.",
            "Quincy Jones has worked with Tim McGraw.",
            "Taylor Swift is an artist.",
            "Quincy Jones has worked with Taylor Swift.",
        ],
    },
]

_DECOMP_INSTRUCTION = (
    "Please break down the following sentence into independent, "
    "self-contained atomic facts. Each fact must be a short, complete "
    "statement that can be understood on its own, without reading any "
    "other fact or the original sentence. If the sentence uses a pronoun "
    "(e.g. \"he\", \"she\", \"his\", \"her\", \"they\", \"their\") that "
    "refers to {entity}, replace the pronoun with \"{entity}\" in the "
    "facts you write. Do not add any information that isn't in the "
    "sentence. Do not add commentary — output only the list of facts."
)


def _build_decomposition_prompt(entity: str, sentence: str) -> str:
    parts = [_DECOMP_INSTRUCTION.format(entity=entity)]
    for demo in _DEMOS:
        parts.append(f"\nSentence: {demo['sentence']}\nIndependent Facts:")
        for fact in demo["facts"]:
            parts.append(f"- {fact}")
    parts.append(f"\nSentence: {sentence}\nIndependent Facts:")
    return "\n".join(parts)


def _parse_facts(raw_output: str) -> list[str]:
    """Parse '- fact' bullet lines out of the model's raw output. Stops at
    the first blank line or a line that doesn't look like a bullet, to
    avoid picking up a stray continuation (e.g. the model starting a new
    'Sentence:' block by mistake)."""
    facts = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            if facts:
                break
            continue
        if line.lower().startswith("sentence:"):
            break
        m = re.match(r"^[-*]\s*(.+)$", line)
        if m:
            fact = m.group(1).strip()
            if fact:
                facts.append(fact)
        elif facts:
            # a non-bullet line after we've already started collecting
            # facts means the model has wandered off-format; stop here
            # rather than risk pulling in commentary as a "fact".
            break
    return facts


@torch.no_grad()
def decompose_sentence(model, tokenizer, entity: str, sentence: str,
                        max_new_tokens: int) -> list[str]:
    prompt = _build_decomposition_prompt(entity, sentence)
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _parse_facts(raw_output)


# ══════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════

def load_done_keys(ckpt_path: str) -> set[int]:
    """question_idx values already fully written to the output file. Since
    each run processes only ONE model_role now (the filename encodes
    which), the checkpoint key is just question_idx, not (question_idx,
    model_role)."""
    done = set()
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["question_idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help="answered_both.jsonl from filter_longform_questions.py")
    parser.add_argument("--model_role", required=True, choices=["teacher", "student"],
                         help="which side's responses to decompose this run. Run this "
                              "script twice (once per role) with two different --output "
                              "paths -- model_role is NOT written into the output "
                              "records, it's meant to be encoded in the output filename "
                              "instead (e.g. atomic_facts_teacher.jsonl).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", type=str, default=None,
                         help="override cfg.decomposition_model_name")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                         help="override cfg.decomposition_max_new_tokens")
    args = parser.parse_args()

    model_name = args.model_name or cfg.decomposition_model_name
    max_new_tokens = args.max_new_tokens or cfg.decomposition_max_new_tokens

    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} entities from {args.input}")
    print(f"model_role: {args.model_role}  ->  output: {args.output}")

    done_idx = load_done_keys(args.output)
    todo = [item for item in items if item["question_idx"] not in done_idx]
    print(f"Checkpoint: {len(done_idx)} question_idx already done, "
          f"{len(todo)} remaining.")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Loading decomposition model: {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map=cfg.device_map)

    response_field = f"{args.model_role}_response"

    with open(args.output, "a", encoding="utf-8") as ckpt_f:
        for item in todo:
            question_idx = item["question_idx"]
            entity = item["entity"]
            response = item[response_field]

            sentences = split_sentences(response)
            print(f"[{question_idx}] {entity}: {len(sentences)} sentence(s)")

            for sentence_idx, sentence in enumerate(sentences):
                facts = decompose_sentence(
                    model, tokenizer, entity, sentence, max_new_tokens
                )
                record = {
                    "question_idx": question_idx,
                    "entity": entity,
                    "sentence_idx": sentence_idx,
                    "sentence": sentence,
                    "facts": facts,
                }
                ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()

    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Wrote to {args.output}")


if __name__ == "__main__":
    main()