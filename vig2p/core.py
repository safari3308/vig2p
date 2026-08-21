from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


VI_FIXUPS: tuple[tuple[str, str], ...] = (
    ("tʃ", "ʧ"),
    ("t̪", "\ue100"),
    ("\ue100", "t"),
    ("e-", "æ"),
    ("1", "→"),
    ("7", "→"),
    ("2", "↘"),
    ("ɜ", "↗"),
    ("3", "↗"),
    ("4", "↓"),
    ("5", "ʔ↗"),
    ("6", "ʔ↓"),
    ("ɗ", "d"),
    ("ʐ", "ʒ"),
    ("̪", ""),
    ("-", ""),
    ("–", "—"),
    ("*", ""),
    ("/", " "),
    ("&", " "),
    ("'", ""),
    ("’", ""),
    ("‘", ""),
    ("đ", "d"),
    ("̩", ""),
)

TEXT_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*|\s+|.", re.UNICODE)
WORD_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*$", re.UNICODE)
VI_MARK_RE = re.compile(r"[À-ỹĐđ]")
NON_VI_S_CLUSTERS = (
    "sc",
    "sh",
    "sk",
    "sl",
    "sm",
    "sn",
    "sp",
    "st",
    "sw",
)


class G2PBackend(Protocol):
    def run(self, text: str):
        ...


def create_backend() -> G2PBackend:
    from ._engine import Pipeline

    return Pipeline()


def _backend_output_to_text(result) -> str:
    return result[0] if isinstance(result, tuple) else str(result)


def tokenize_text(text: str) -> list[str]:
    text = text.replace("’", "'").replace("‘", "'")
    return TEXT_TOKEN_RE.findall(text)

VI_UNMARKED_WORDS = {
    # Default unmarked words
    "do", "to", "so", "no", "ta", "va", "ra", "xa", "ma", "cho",
    # Short words that match English words
    "me", "he", "an", "am", "on", "go", "my", "be",
    "can", "man", "fan", "ban", "van", "tan", "tin", "pin",
    "bat", "cat", "mat", "bit", "fit", "hit", "cut", "put",
    # Popular Vietnamese words without marks
    "tam", "tui", "tuy", "suy", "sang", "song", "tra", "tri", "tru"
}
# 🌟 Phoneme lookup table for Vietnamese words without marks that are easily confused with English
UNMARKED_PHONEME_MAP = {
    "tu": "tu",
    "to": "tɔ",
    "do": "zɔ",
    "so": "sɔ",
    "no": "nɔ",
    "tam": "ta:m",
    "tin": "tin",
    "tan": "ta:n",
}

EN_ONLY_CHARS = set("fjwz")
EN_CONSONANT_CLUSTERS = (
    "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw",
    "cl", "fl", "gl", "pl", "br", "cr", "dr", "fr", "gr", "pr",
    "str", "spr", "spl", "sch", "thr",
)
EN_SUFFIXES = (
    "tion", "sion", "ing", "ment", "able", "ible", "ness", "less",
    "full", "fully", "ed", "ce", "cy", "ty", "'s", "'t", "'re", "'ve", "'ll", "'d", "ck"
)
UNAMBIGUOUS_EN_WORDS = {
    "the", "and", "is", "are", "you", "they", "we", "he", "she", "it",
    "this", "that", "with", "from", "for", "not", "but", "have", "has", "had",
    "what", "where", "when", "why", "how", "all", "any", "your", "my", "her",
    "his", "our", "their", "will", "would", "can", "could", "should", "don't",
    "doesn't", "didn't", "won't", "can't", "about", "like", "just", "over",
    "into", "than", "them", "some", "time", "very", "come", "here", "make",
    "even", "back", "good", "give", "most", "only", "also", "then", "look",
    "more", "day", "way", "well", "new", "want", "because", "these", "user",
    "app", "test", "build", "code", "dev", "bug", "log", "start", "stop",
    "click", "run", "set", "get", "post", "delete", "data", "file", "page",
    "site", "web", "server", "system", "service", "status", "error", "team",
    "style", "travel", "giant", "budget", "panic", "slow", "fast", "easy",
    "check", "in", "fan", "page"
}
SENTENCE_STOP_PUNCT = {".", "!", "?", ";", ":"}


DEFINITE_VI_UNMARKED_WORDS = {
    "ta", "va", "ra", "xa", "ma", "cho", "tam", "tui", "tuy", "suy",
    "sang", "song", "tra", "tri", "tru", "tin", "tan"
}
SHARED_AMBIGUOUS_WORDS = {
    "do", "to", "so", "no", "me", "he", "an", "am", "on", "go", "my", "be",
    "can", "man", "ban", "van", "tan", "tin", "pin",
    "bat", "cat", "mat", "bit", "fit", "hit", "cut", "put",
    "at", "it", "is", "or", "if", "us", "up"
}


def _is_definite_vi(word: str) -> bool:
    if VI_MARK_RE.search(word) is not None:
        return True
    lower = word.lower()
    return lower in DEFINITE_VI_UNMARKED_WORDS


def _is_definite_en(word: str) -> bool:
    lower = word.lower()
    if lower in SHARED_AMBIGUOUS_WORDS:
        return False
    if lower in UNAMBIGUOUS_EN_WORDS:
        return True
    if any(c in EN_ONLY_CHARS for c in lower):
        return True
    if any(lower.startswith(cluster) for cluster in EN_CONSONANT_CLUSTERS):
        return True
    if len(lower) > 3 and any(lower.endswith(suf) for suf in EN_SUFFIXES):
        return True
    return False


def classify_token_languages(tokens: list[str]) -> list[str | None]:
    token_count = len(tokens)
    base_langs: list[str | None] = [None] * token_count

    for i, token in enumerate(tokens):
        if not WORD_RE.match(token):
            continue
        if _is_definite_vi(token):
            base_langs[i] = "vi"
        elif _is_definite_en(token):
            base_langs[i] = "en"
        else:
            base_langs[i] = "ambiguous"

    final_langs: list[str | None] = list(base_langs)

    for i, token in enumerate(tokens):
        if base_langs[i] != "ambiguous":
            continue

        left_vi_dist = 999
        left_en_dist = 999
        for left in range(i - 1, -1, -1):
            if tokens[left] in SENTENCE_STOP_PUNCT:
                break
            if base_langs[left] == "vi":
                left_vi_dist = i - left
                break
            elif base_langs[left] == "en":
                left_en_dist = i - left
                break

        right_vi_dist = 999
        right_en_dist = 999
        for right in range(i + 1, token_count):
            if tokens[right] in SENTENCE_STOP_PUNCT:
                break
            if base_langs[right] == "vi":
                right_vi_dist = right - i
                break
            elif base_langs[right] == "en":
                right_en_dist = right - i
                break

        min_vi = min(left_vi_dist, right_vi_dist)
        min_en = min(left_en_dist, right_en_dist)

        if min_vi < min_en:
            final_langs[i] = "vi"
        elif min_en < min_vi:
            final_langs[i] = "en"
        else:
            if left_en_dist < left_vi_dist:
                final_langs[i] = "en"
            elif left_vi_dist < left_en_dist:
                final_langs[i] = "vi"
            elif right_en_dist < right_vi_dist:
                final_langs[i] = "en"
            elif right_vi_dist < right_en_dist:
                final_langs[i] = "vi"
            elif left_vi_dist < 999 or right_vi_dist < 999:
                final_langs[i] = "vi"
            elif left_en_dist < 999 or right_en_dist < 999:
                final_langs[i] = "en"
            else:
                final_langs[i] = "vi"

    return final_langs


def fix_phonemes(
    phonemes: str,
    source_text: str | None = None,
    *,
    preserve_unmarked_vietnamese_onsets: bool = True,
    is_vietnamese: bool | None = None,
) -> str:
    for old, new in VI_FIXUPS:
        phonemes = phonemes.replace(old, new)

    if source_text:
        source_lower = source_text.lower()

        if is_vietnamese is None:
            is_vi = (VI_MARK_RE.search(source_text) is not None) or (source_lower in VI_UNMARKED_WORDS)
        else:
            is_vi = is_vietnamese

        if is_vi:
            # 🌟 1. Prioritize dictionary lookup (O(1) lookup - super fast and concise)
            if source_lower in UNMARKED_PHONEME_MAP:
                return UNMARKED_PHONEME_MAP[source_lower]

            # 🌟 2. Handle general logic for remaining words
            preserve_unmarked = preserve_unmarked_vietnamese_onsets
            if source_lower.startswith("th"):
                # 1. If G2P outputs 'θ' -> convert to 't h'
                if "θ" in phonemes:
                    phonemes = phonemes.replace("θ", "t h", 1)
                # 2. If G2P outputs 't' (stuck in words like thi..., thiên, thịt...) -> add 'h' after 't'
                elif phonemes.startswith("t") and not phonemes.startswith("t h"):
                    phonemes = "t h" + phonemes[1:]
            elif source_lower.startswith("tr"):
                phonemes = phonemes.replace("ʧ", "ʈʂ", 1)
            elif (
                (is_vi or preserve_unmarked)
                and source_lower.startswith("s")
                and not source_lower.startswith(NON_VI_S_CLUSTERS)
            ):
                phonemes = phonemes.replace("s", "ʂ", 1)
            elif source_lower.startswith("gi") or re.match(r"^g[iìíỉĩị]", source_lower):
                phonemes = phonemes.replace("z", "ʝ", 1)
    return phonemes


def phonemize_text(
    text: str,
    backend: G2PBackend | None = None,
    *,
    preserve_unmarked_vietnamese_onsets: bool = True,
) -> str:
    backend = backend or create_backend()
    tokens = tokenize_text(text)
    token_langs = classify_token_languages(tokens)
    pieces: list[str] = []

    for token, lang in zip(tokens, token_langs):
        if token.isspace():
            pieces.append(" ")
        elif WORD_RE.match(token):
            raw = _backend_output_to_text(backend.run(token))
            is_vi = (lang == "vi") if lang is not None else None
            pieces.append(
                fix_phonemes(
                    raw,
                    source_text=token,
                    preserve_unmarked_vietnamese_onsets=preserve_unmarked_vietnamese_onsets,
                    is_vietnamese=is_vi,
                )
            )
        else:
            pieces.append(fix_phonemes(token))
    return "".join(pieces).strip()


def phonemize_many(
    texts: Iterable[str],
    backend: G2PBackend | None = None,
    *,
    preserve_unmarked_vietnamese_onsets: bool = True,
) -> list[str]:
    backend = backend or create_backend()
    return [
        phonemize_text(
            text,
            backend=backend,
            preserve_unmarked_vietnamese_onsets=preserve_unmarked_vietnamese_onsets,
        )
        for text in texts
    ]


@dataclass
class VietnameseG2P:
    backend: G2PBackend | None = None
    preserve_unmarked_vietnamese_onsets: bool = True

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = create_backend()

    def __call__(self, text: str) -> str:
        return phonemize_text(
            text,
            backend=self.backend,
            preserve_unmarked_vietnamese_onsets=self.preserve_unmarked_vietnamese_onsets,
        )

    def many(self, texts: Iterable[str]) -> list[str]:
        return phonemize_many(
            texts,
            backend=self.backend,
            preserve_unmarked_vietnamese_onsets=self.preserve_unmarked_vietnamese_onsets,
        )

