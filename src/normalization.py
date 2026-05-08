"""Text normalization for taxon entity retrieval."""

from __future__ import annotations

import re
import unicodedata


LATIN_ABBREVIATION_RE = re.compile(r"\b([A-Za-z])\s*\.\s*([A-Za-z][A-Za-z-]*)\b")
WHITESPACE_RE = re.compile(r"\s+")
EDGE_PUNCTUATION_RE = re.compile(r"^[\s,;:，；：、]+|[\s,;:，；：、]+$")


def normalize_text(text: str) -> str:
    """Normalize user and catalog names for exact matching and lookup."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("。", ".")
    normalized = normalized.replace("．", ".")
    normalized = LATIN_ABBREVIATION_RE.sub(_format_latin_abbreviation, normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = EDGE_PUNCTUATION_RE.sub("", normalized)
    return normalized.casefold()


def display_normalized_text(text: str) -> str:
    """Normalize formatting while preserving display-oriented casing."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("。", ".")
    normalized = normalized.replace("．", ".")
    normalized = LATIN_ABBREVIATION_RE.sub(_format_latin_abbreviation, normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return EDGE_PUNCTUATION_RE.sub("", normalized)


def is_latin_abbreviation(text: str) -> bool:
    """Return whether the normalized text is exactly like 'E. coli'."""
    normalized = normalize_text(text)
    return re.fullmatch(r"[a-z]\. [a-z][a-z-]*", normalized) is not None


def parse_latin_abbreviation(text: str) -> tuple[str, str] | None:
    """Parse an abbreviation into (genus_initial, species_epithet)."""
    normalized = normalize_text(text)
    match = re.fullmatch(r"([a-z])\. ([a-z][a-z-]*)", normalized)
    if not match:
        return None
    return match.group(1), match.group(2)


def _format_latin_abbreviation(match: re.Match[str]) -> str:
    initial = match.group(1).upper()
    epithet = match.group(2).lower()
    return f"{initial}. {epithet}"
