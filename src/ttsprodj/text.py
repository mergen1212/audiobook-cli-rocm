"""Нормализация русского текста и разбиение на фрагменты для синтеза речи."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from num2words import num2words


@dataclass(frozen=True)
class Chunk:
    text: str
    end_of_paragraph: bool


_DASH_START_RE = re.compile(r"^\s*[—–-]\s*")
_DASH_RE = re.compile(r"\s*[—–]\s*|\s+-\s+")
_QUOTES_RE = re.compile(r'[«»„“”"‘’]')
_SYMBOLS = {"%": " процентов", "№": "номер ", "§": "параграф ", "&": " и ", "/": " ", "*": " "}
_GROUPED_NUMBER_RE = re.compile(r"(?<!\d)\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?…:;])")
_DOUBLE_PUNCT_RE = re.compile(r"([,;:])\s*[,;:]")
_SPACES_RE = re.compile(r"\s+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")


def _number_to_words(match: re.Match[str]) -> str:
    raw = match.group(0).replace(",", ".")
    value = float(raw) if "." in raw else int(raw)
    try:
        return num2words(value, lang="ru")
    except (OverflowError, NotImplementedError):
        return match.group(0)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _DASH_START_RE.sub("", text)
    text = _QUOTES_RE.sub("", text)
    text = _DASH_RE.sub(", ", text)
    for symbol, replacement in _SYMBOLS.items():
        text = text.replace(symbol, replacement)
    text = _GROUPED_NUMBER_RE.sub(lambda m: re.sub(r"\D", "", m.group(0)), text)
    text = _NUMBER_RE.sub(_number_to_words, text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _DOUBLE_PUNCT_RE.sub(r"\1", text)
    text = _SPACES_RE.sub(" ", text).strip()
    if text.endswith((",", ";", ":")):
        text = text[:-1] + "."
    elif text and not text.endswith((".", "!", "?", "…")):
        text += "."
    return text


def split_into_chunks(paragraphs: list[str], max_chars: int = 400) -> list[Chunk]:
    if max_chars < 50:
        raise ValueError("max_chars должен быть не меньше 50")
    chunks: list[Chunk] = []
    for paragraph in paragraphs:
        text = normalize_text(paragraph)
        if not text:
            continue
        pieces = _pack(_split_sentences(text, max_chars), max_chars)
        last = len(pieces) - 1
        chunks.extend(Chunk(piece, end_of_paragraph=i == last) for i, piece in enumerate(pieces))
    return chunks


def _split_sentences(text: str, max_chars: int) -> list[str]:
    sentences: list[str] = []
    for sentence in _SENTENCE_END_RE.split(text):
        sentences.extend(_split_long(sentence, max_chars))
    return sentences


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}" if current else sentence
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
