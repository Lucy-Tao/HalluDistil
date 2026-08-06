"""
abstention_rule.py -- single source of truth for HalluDistil's long-form
abstention / non-answer detection.

Both compute_abstention_rate.py (which reports the abstention RATE) and the
FActScore evaluation script (which BLANKS abstained responses to "") import
`is_abstained` from here, so the two stages can never drift apart.

Rule = merged phrase list (SHARS non-answer phrases + the extra variants we
found missing, e.g. "no well-known", "not a widely recognized") PLUS a
position check: a phrase only counts as an abstention if it appears within
the first CUTOFF_CHARS characters. The position check is what stops a
mid-text mention like "the fish is not widely known" (Botak Chin, char 542)
from being misread as a refusal, while still catching genuine refusals that
open with "There is no widely known ...".

hedging phrases ("multiple individuals", "several individuals", "common
name", "can refer to") are INCLUDED -- per the finalized decision they count
as non-answers.
"""

CUTOFF_CHARS = 200

MERGED_PATTERNS = sorted(set([
    # refusal / not-found
    "no widely known", "no widely recognized", "not widely known",
    "not widely recognized", "not a widely known", "not a widely recognized",
    "no well-known", "no well known", "not a well-known", "not well-known",
    "not known", "no known", "no prominent", "no notable", "no famously known",
    "not widely available", "not publicly available", "not publicly",
    "no publicly known", "no publicly available", "no publicly documented",
    "not publicly documented", "does not exist", "not a real person",
    "no verified biography", "no verified information", "no verified record",
    "no information available", "no prominent record", "no prominent information",
    # first-person disclaimers
    "i could not find", "couldn't find", "i'm not aware", "i am not aware",
    "i do not know", "i don't know", "i'm not sure", "i cannot confirm",
    "i cannot guarantee", "i cannot verify", "i do not have information",
    "i don't have information", "i have no information", "i lack information",
    # hedging (counted as non-answers per decision)
    "common name", "can refer to", "several individuals",
    "multiple individuals", "few individuals",
    # soft disclaimers
    "not available", "cannot guarantee", "recommend checking",
    "may want to check", "please check", "consider checking",
    "little information", "further clarification may help",
    "may not be accurate", "information may be incomplete",
    "information may not be reliable", "details are limited",
]))


def abstention_hit(response, cutoff_chars=CUTOFF_CHARS):
    """Return (is_abstained: bool, phrase: str|None, position: int|None).

    Finds the EARLIEST-matching phrase, then applies the position check.
    Empty / non-string input counts as abstained.
    """
    if not response or not isinstance(response, str):
        return True, "<empty>", 0
    low = response.strip().lower()
    best = None
    for pat in MERGED_PATTERNS:
        idx = low.find(pat)
        if idx != -1 and (best is None or idx < best[1]):
            best = (pat, idx)
    if best is None:
        return False, None, None
    return (best[1] < cutoff_chars), best[0], best[1]


def is_abstained(response, cutoff_chars=CUTOFF_CHARS):
    """Convenience boolean wrapper around abstention_hit()."""
    return abstention_hit(response, cutoff_chars)[0]