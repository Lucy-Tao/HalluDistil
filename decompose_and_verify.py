"""
decompose_and_verify.py — Phase 3: for ONE model_role (teacher / student /
distilled_student), decompose every response in answered_both.jsonl into
factual claims and verify each claim's correctness via a web-search-
grounded LLM judge, using the OpenAI Responses API (directly, or through
Oxford's Lagrange gateway).

Checkpointed incrementally per question_idx — safe to resume after a
timeout/crash. Each run processes ONE model_role; run it separately per
role (teacher, student, and later distilled_student), with a different
--output path each time so results don't overwrite each other -- same
"one role per run, role encoded in filename not in the record" pattern
as compute_abstention_rate.py / decompose_atomic_facts.py.

Usage
-----
  python decompose_and_verify.py \\
      --input gen_longform_data/answered_both.jsonl \\
      --model_role teacher \\
      --output gen_longform_data/claims_teacher.jsonl \\
      --model gpt-5-mini \\
      --api_key $OPENAI_API_KEY

  python decompose_and_verify.py \\
      --input gen_longform_data/answered_both.jsonl \\
      --model_role student \\
      --output gen_longform_data/claims_student.jsonl \\
      --model gpt-5-mini \\
      --api_key $OPENAI_API_KEY

  # Through Oxford's Lagrange gateway instead of OpenAI directly:
  python decompose_and_verify.py \\
      --input gen_longform_data/answered_both.jsonl \\
      --model_role teacher \\
      --output gen_longform_data/claims_teacher.jsonl \\
      --model gpt-5-mini \\
      --api_key $LAGRANGE_API_KEY \\
      --base_url https://lagrange.uksouth.cloudapp.azure.com/openai

  # Later, for a distilled student: point --input at a file with the same
  # {question_idx, entity, distilled_student_response} schema (e.g. built
  # by re-running filter_longform_questions.py-style logic against the
  # distilled student's own generations), and --model_role distilled_student.

Output: one jsonl line per (question_idx, claim_idx) for the chosen
model_role:
  {
    "question_idx": int,
    "entity": str,
    "claim_idx": int,
    "claim": str,
    "is_true": bool,
  }

Requires: pip install --upgrade openai --break-system-packages
"""
import argparse
import json
import os
import re

DECOMP_INSTRUCTION = (
    "Please list the specific factual propositions included in the answer "
    "above. Be complete and do not leave any factual claims out. Do not "
    "list the same fact more than once, even if it is phrased "
    "differently elsewhere in the answer. Provide each claim as a "
    "separate sentence in a separate bullet point. For each claim, use "
    "web search to check it against current, reliable sources, then "
    "immediately after the claim append its correctness in square "
    "brackets as either [True] or [False]. Format: "
    "\"- <claim>. [True]\" or \"- <claim>. [False]\"."
)

_CLAIM_LINE = re.compile(r"^[-*]\s*(.+?)\s*\[(True|False)\]", re.IGNORECASE)


def parse_claims(raw_output: str) -> list[dict]:
    claims = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _CLAIM_LINE.match(line)
        if m:
            claims.append({
                "claim": m.group(1).strip(),
                "is_true": m.group(2).lower() == "true",
            })
    return claims


def decompose_and_verify(client, model: str, question_prompt: str, answer: str) -> list[dict]:
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": DECOMP_INSTRUCTION},
        ],
    )
    return parse_claims(response.output_text)


def load_done_idx(ckpt_path: str) -> set[int]:
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
                         help="answered_both.jsonl (or, for a distilled "
                              "student, a file with the same schema)")
    parser.add_argument("--model_role", required=True,
                         help="which side's responses to decompose this "
                              "run, e.g. 'teacher', 'student', or later "
                              "'distilled_student'. Reads the "
                              "'{model_role}_response' field from --input. "
                              "NOT written into output records -- encode it "
                              "in --output's filename instead.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5-mini",
                         help="OpenAI (or gateway) model to use")
    parser.add_argument("--api_key", default=None,
                         help="API key. If omitted, reads from the "
                              "OPENAI_API_KEY environment variable.")
    parser.add_argument("--base_url", default=None,
                         help="Override the API base URL, e.g. Oxford's "
                              "Lagrange gateway: "
                              "https://lagrange.uksouth.cloudapp.azure.com/openai "
                              "(requires being on the University network). "
                              "If omitted, uses OpenAI's default endpoint.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Pass --api_key or set the OPENAI_API_KEY "
            "environment variable."
        )

    with open(args.input, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} entities from {args.input}")
    print(f"model_role: {args.model_role}  ->  output: {args.output}")

    response_field = f"{args.model_role}_response"
    missing_field = [it for it in items if response_field not in it]
    if missing_field:
        raise SystemExit(
            f"--input is missing the '{response_field}' field on "
            f"{len(missing_field)} record(s) -- check --model_role matches "
            f"the schema of --input (e.g. answered_both.jsonl has "
            f"'teacher_response' and 'student_response', not "
            f"'{response_field}')."
        )

    done_idx = load_done_idx(args.output)
    todo = [it for it in items if it["question_idx"] not in done_idx]
    print(f"Checkpoint: {len(done_idx)} question_idx already done, "
          f"{len(todo)} remaining.")

    if not todo:
        print("Nothing to do.")
        return

    from openai import OpenAI
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    with open(args.output, "a", encoding="utf-8") as ckpt_f:
        for item in todo:
            question_idx = item["question_idx"]
            entity = item["entity"]
            question_prompt = item.get(
                "prompt", f"Question: Tell me a bio of {entity}."
            )
            answer = item[response_field]

            print(f"[{question_idx}] {entity}")
            try:
                claims = decompose_and_verify(client, args.model, question_prompt, answer)
            except Exception as e:  # noqa: BLE001 -- log and keep going; rerun will retry this question_idx
                print(f"  ERROR on question_idx={question_idx} ({entity!r}): {e}")
                print(f"  Skipping for now -- rerun this script to retry "
                      f"(checkpoint will pick it back up).")
                continue

            if not claims:
                print(f"  WARNING: 0 claims parsed for {entity!r} -- check "
                      f"raw output format didn't drift. Not writing a "
                      f"checkpoint entry, so this will be retried on rerun.")
                continue

            for claim_idx, c in enumerate(claims):
                record = {
                    "question_idx": question_idx,
                    "entity": entity,
                    "claim_idx": claim_idx,
                    "claim": c["claim"],
                    "is_true": c["is_true"],
                }
                ckpt_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            print(f"  -> {len(claims)} claims "
                  f"({sum(c['is_true'] for c in claims)} True, "
                  f"{sum(not c['is_true'] for c in claims)} False)")

    print(f"\nDone. Wrote to {args.output}")


if __name__ == "__main__":
    main()