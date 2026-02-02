"""Object name extraction from LIBERO task instruction strings.

Uses regex patterns matched against common LIBERO task phrasings. Falls back
to simple noun-phrase extraction for unrecognized patterns.
"""

from __future__ import annotations

import re
from typing import List


# Ordered list of extraction patterns. Each entry is (pattern, group_indices)
# where group_indices lists which regex groups contain object names.
_PATTERNS: List[re.Pattern] = []


def _p(pattern: str) -> re.Pattern:
    compiled = re.compile(pattern, re.IGNORECASE)
    _PATTERNS.append(compiled)
    return compiled


# "pick up the X and place it in/on the Y"
_p(r"pick up (?:the )?(.+?) and place (?:it )?(?:in|on|into|onto) (?:the )?(.+?)$")

# "put the X in/on the Y"
_p(r"put (?:the )?(.+?) (?:in|on|into|onto) (?:the )?(.+?)$")

# "put both the X and the Y in the Z"
_p(r"put both (?:the )?(.+?) and (?:the )?(.+?) (?:in|on|into|onto) (?:the )?(.+?)$")

# "place the X in/on the Y"
_p(r"place (?:the )?(.+?) (?:in|on|into|onto) (?:the )?(.+?)$")

# "pick up the X"
_p(r"pick up (?:the )?(.+?)$")

# "open the X"
_p(r"open (?:the )?(.+?)$")

# "close the X"
_p(r"close (?:the )?(.+?)$")

# "push the X"
_p(r"push (?:the )?(.+?)$")

# "turn on the X"
_p(r"turn on (?:the )?(.+?)$")

# "turn off the X"
_p(r"turn off (?:the )?(.+?)$")

# "stack the X on top of the Y"
_p(r"stack (?:the )?(.+?) on top of (?:the )?(.+?)$")

# "move the X to the Y"
_p(r"move (?:the )?(.+?) to (?:the )?(.+?)$")


def extract_object_names(task_instruction: str) -> List[str]:
    """Extract object names from a LIBERO task instruction.

    Args:
        task_instruction: Natural language task description,
            e.g. ``"pick up the red block and place it in the blue bowl"``.

    Returns:
        List of extracted object name strings (deduplicated, order preserved).
    """
    instruction = task_instruction.strip()

    for pattern in _PATTERNS:
        match = pattern.search(instruction)
        if match:
            names = [g.strip() for g in match.groups() if g]
            return _deduplicate(names)

    # Fallback: split on common prepositions and extract trailing noun phrases
    return _fallback_extract(instruction)


def _fallback_extract(instruction: str) -> List[str]:
    """Best-effort extraction when no pattern matches."""
    # Remove common verbs/articles and split on prepositions
    cleaned = instruction.lower()
    for verb in ["pick up", "put", "place", "push", "open", "close",
                 "turn on", "turn off", "move", "stack", "grasp", "grab"]:
        cleaned = cleaned.replace(verb, "")

    # Split on prepositions
    parts = re.split(r"\b(?:the|in|on|into|onto|to|from|with|and|it|both)\b", cleaned)
    names = [p.strip() for p in parts if p.strip()]
    return _deduplicate(names)


def _deduplicate(names: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: List[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result
