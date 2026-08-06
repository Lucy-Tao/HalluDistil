"""
test_llm_decompose.py — quick one-shot test of whole-response claim
decomposition + web-search-grounded correctness judgment via the OpenAI
API, combining:
  - the semantic-entropy-paper-style whole-response decomposition
    instruction (NOT FActScore's sentence-by-sentence approach)
  - a no-duplicates requirement
  - an inline correctness verdict per claim, grounded via OpenAI's
    web_search tool (Responses API) rather than the model's own
    unverified parametric knowledge

This is a TEST script only — decomposes ONE entity's response and prints
the raw result, so you can eyeball quality before this logic gets folded
into the real pipeline script (checkpointing, running over every entity
for both teacher and student, etc.).

Uses the Responses API (client.responses.create), NOT the older Chat
Completions API (client.chat.completions.create) — the web_search tool
is only available there. The decomposition instruction is issued as a
follow-up turn in a simulated conversation (question -> the
already-generated answer -> decomposition request), so "the answer
above" has a real referent for the model.

Requires: pip install --upgrade openai --break-system-packages
Requires: an OpenAI API key, either passed via --api_key or set as the
OPENAI_API_KEY environment variable.

Usage
-----
  python test_llm_decompose.py \\
      --input ~/SimpleQA/gen_longform_data/gen_factscore_bio_Qwen3-14B.jsonl \\
      --entity "Roberto Clemente" \\
      --model gpt-5-mini
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

# Deliberately NOT instructing the model to omit source citations/links
# after the verdict -- testing showed it doesn't reliably change behavior
# (the built-in web_search tool keeps appending citations regardless).
# Parsing below already tolerates and discards whatever trails the
# verdict, so that instruction added prompt length for zero functional
# benefit -- removed rather than kept as a no-op.

# Parses lines like "- Roberto Clemente was born in 1934. [True]" and also
# tolerates trailing content after the verdict, e.g. a citation the model
# appends: "- ...claim. [True]. ([baseballhall.org](https://...))"
# — NOT anchored to end-of-line (no trailing $), since requiring that
# caused every single claim to fail to parse in initial testing (gpt-5-mini
# reliably appends a source citation after the [True]/[False] bracket).
_CLAIM_LINE = re.compile(r"^[-*]\s*(.+?)\s*\[(True|False)\]", re.IGNORECASE)


def parse_claims(raw_output: str) -> list[dict]:
    """Parse '- claim text [True/False]' lines into structured records.
    Lines that don't match the expected format (e.g. the model added
    stray commentary) are skipped, not silently mis-parsed — check
    len(parsed) against the number of bullet-looking lines in raw_output
    if you suspect something was dropped."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                         help="generation jsonl (from generate_longform_responses.py)")
    parser.add_argument("--entity", required=True,
                         help="exact entity name to test (must match an 'entity' "
                              "field in --input)")
    parser.add_argument("--model", default="gpt-5-mini",
                         help="OpenAI model to use (must support the web_search "
                              "tool in the Responses API)")
    parser.add_argument("--api_key", default=None,
                         help="OpenAI API key. If omitted, reads from the "
                              "OPENAI_API_KEY environment variable.")
    parser.add_argument("--base_url", default=None,
                         help="Override the API base URL. Use this if you're "
                              "going through a gateway instead of OpenAI "
                              "directly -- e.g. Oxford's Lagrange gateway: "
                              "https://lagrange.uksouth.cloudapp.azure.com/openai "
                              "(requires being on the University network: "
                              "physical connection, University WiFi, or VPN). "
                              "If omitted, uses OpenAI's default endpoint.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key found. Pass --api_key or set the OPENAI_API_KEY "
            "environment variable."
        )

    # Find the requested entity's response
    target = None
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["entity"] == args.entity:
                target = rec
                break
    if target is None:
        raise SystemExit(f"Entity {args.entity!r} not found in {args.input}")

    question_prompt = target.get("prompt", f"Question: Tell me a bio of {args.entity}.")
    answer = target["response"]

    print("=" * 70)
    print(f"Entity: {args.entity}")
    print("=" * 70)
    print(f"\n--- Question ---\n{question_prompt}")
    print(f"\n--- Answer (being decomposed) ---\n{answer}")

    from openai import OpenAI
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    response = client.responses.create(
        model=args.model,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": DECOMP_INSTRUCTION},
        ],
    )
    raw_output = response.output_text

    print(f"\n--- Raw decomposition (model: {args.model}) ---")
    print(raw_output)

    claims = parse_claims(raw_output)
    n_bullet_lines = sum(1 for l in raw_output.split("\n") if l.strip().startswith(("-", "*")))
    print(f"\n--- Parsed: {len(claims)} claims (saw {n_bullet_lines} bullet-looking lines "
          f"in raw output -- mismatch means some lines didn't match the expected format) ---")
    for c in claims:
        print(f"  [{'True ' if c['is_true'] else 'False'}] {c['claim']}")

    # Surface which queries were actually searched, for a sanity check
    # that grounding is really happening and not just cosmetic
    search_calls = [item for item in response.output if getattr(item, "type", None) == "web_search_call"]
    print(f"\n--- {len(search_calls)} web_search_call(s) made ---")
    for call in search_calls:
        action = getattr(call, "action", None)
        query = getattr(action, "query", None) if action else None
        if query:
            print(f"  searched: {query!r}")

    # Token usage, if available
    usage = getattr(response, "usage", None)
    if usage:
        print(f"\n--- Token usage ---")
        print(f"input_tokens: {usage.input_tokens}, "
              f"output_tokens: {usage.output_tokens}, "
              f"total: {usage.total_tokens}")


if __name__ == "__main__":
    main()