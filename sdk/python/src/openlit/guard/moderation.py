"""
Content moderation guard -- profanity & toxicity detection.

Uses local keyword/regex patterns.  Phases: preflight + postflight.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional, Tuple

from openlit.guard._base import Guard, GuardPhase, GuardResult

# A compact set of commonly flagged English profanity.  This is NOT
# exhaustive -- users can extend via ``custom_words``.  Each word is
# matched as a whole word (\\b boundary) case-insensitively.
_BUILTIN_WORDS: List[str] = [
    "fuck",
    "shit",
    "ass",
    "bitch",
    "damn",
    "bastard",
    "crap",
    "dick",
    "piss",
    "cock",
    "cunt",
    "slut",
    "whore",
    "nigger",
    "nigga",
    "faggot",
    "retard",
    "kike",
    "spic",
]

_TOXICITY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:kill\s+yourself|kys|go\s+die|hope\s+you\s+die)\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'ll| will)\s+(?:kill|hurt|destroy)\s+you)\b", re.IGNORECASE),
]

# Single alternation of the toxicity patterns, used only for redaction.  The
# per-pattern list above is kept intact so classification/score detection
# retains its existing semantics.
_TOXICITY_RE = re.compile(
    "|".join(f"(?:{pattern.pattern})" for pattern in _TOXICITY_PATTERNS),
    re.IGNORECASE,
)


def _build_profanity_pattern(words: List[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in words]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def _redact_spans(text: str, profanity_re: re.Pattern) -> str:
    spans: List[Tuple[int, int, str]] = []
    for match in _TOXICITY_RE.finditer(text):
        spans.append((match.start(), match.end(), "[REDACTED:toxicity]"))

    toxic_spans = [(start, end) for start, end, _ in spans]
    for match in profanity_re.finditer(text):
        start, end = match.span()
        if any(start < toxic_end and end > toxic_start for toxic_start, toxic_end in toxic_spans):
            continue
        spans.append((start, end, "[REDACTED:profanity]"))

    spans.sort(key=lambda span: span[0])
    parts: List[str] = []
    cursor = 0
    for start, end, replacement in spans:
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


class Moderation(Guard):
    """
    Detects profanity and toxic language.

    Parameters
    ----------
    action : str
        ``"warn"`` (default), ``"deny"``, or ``"redact"``.
    custom_words : list[str], optional
        Additional words to flag.
    """

    name = "moderation"
    phases = (GuardPhase.PREFLIGHT, GuardPhase.POSTFLIGHT)

    def __init__(
        self,
        action: str = "warn",
        custom_words: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(action=action, **kwargs)
        all_words = _BUILTIN_WORDS + (custom_words or [])
        self._profanity_re = _build_profanity_pattern(all_words)

    def run(self, text: str, phase: GuardPhase) -> GuardResult:
        result = super().run(text, phase)
        if result.transformed_text is None or not self._max_scan_length:
            return result
        suffix = text[self._max_scan_length :]
        if not suffix:
            return result
        return replace(result, transformed_text=result.transformed_text + suffix)

    def evaluate(self, text: str) -> GuardResult:
        profanity_matches = self._profanity_re.findall(text)
        toxicity_hits = any(pat.search(text) for pat in _TOXICITY_PATTERNS)

        if not profanity_matches and not toxicity_hits:
            return GuardResult(guard_name=self.name)

        classifications: List[str] = []
        if profanity_matches:
            classifications.append("profanity")
        if toxicity_hits:
            classifications.append("toxicity")

        classification = ", ".join(classifications)
        score = 0.7 if not toxicity_hits else 0.95

        transformed: Optional[str] = None
        if self._action.value == "redact":
            transformed = _redact_spans(text, self._profanity_re)

        return GuardResult(
            action=self._action,
            score=score,
            guard_name=self.name,
            classification=classification,
            explanation=f"Moderation flag: {classification}",
            transformed_text=transformed,
        )
