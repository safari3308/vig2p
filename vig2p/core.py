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

VI_UNMARKED_WORDS = {"do", "to", "so", "no", "ta", "va", "ra", "xa", "ma", "cho"}
# 🌟 Bảng tra cứu Phoneme chuẩn cho các từ tiếng Việt không dấu dễ bị nhận nhầm sang tiếng Anh
UNMARKED_PHONEME_MAP = {
    "tu": "t u 1",
    "to": "t o 1",
    "do": "z o 1",   # Tiếng Việt miền Bắc 'd' là âm 'z' (zo/dô)
    "so": "s o 1",
    "no": "n o 1",
    "ta": "t a 1",
    "va": "v a 1",
    "ra": "r a 1",
    "xa": "x a 1",
    "ma": "m a 1",
    "cho": "c ɔ 1",
}

def fix_phonemes(
    phonemes: str,
    source_text: str | None = None,
    *,
    preserve_unmarked_vietnamese_onsets: bool = True,
) -> str:
    for old, new in VI_FIXUPS:
        phonemes = phonemes.replace(old, new)

    if source_text:
        source_lower = source_text.lower()

        # 🌟 1. Ưu tiên tra bảng Dictionary (O(1) lookup - siêu nhanh và ngắn gọn)
        if source_lower in UNMARKED_PHONEME_MAP:
            return UNMARKED_PHONEME_MAP[source_lower]

        # 🌟 2. Xử lý logic chung cho các từ còn lại
        has_vi_mark = (VI_MARK_RE.search(source_text) is not None) or (source_lower in VI_UNMARKED_WORDS)

        preserve_unmarked = preserve_unmarked_vietnamese_onsets
        if source_lower.startswith("th"):
            phonemes = phonemes.replace("θ", "t", 1)
        elif source_lower.startswith("tr"):
            phonemes = phonemes.replace("ʧ", "ʈʂ", 1)
        elif (
            (has_vi_mark or preserve_unmarked)
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
    pieces: list[str] = []
    for token in tokenize_text(text):
        if token.isspace():
            pieces.append(" ")
        elif WORD_RE.match(token):
            raw = _backend_output_to_text(backend.run(token))
            pieces.append(
                fix_phonemes(
                    raw,
                    source_text=token,
                    preserve_unmarked_vietnamese_onsets=preserve_unmarked_vietnamese_onsets,
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
