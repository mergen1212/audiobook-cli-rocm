"""Чтение книг (TXT, FB2, EPUB) и разбиение их на главы."""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

SUPPORTED_SUFFIXES = (".txt", ".fb2", ".epub")

_TXT_HEADING_RE = re.compile(
    r"^(глава|часть|пролог|эпилог)\b(\s+[\w-]+)?[.:]?(\s.{0,80})?$", re.IGNORECASE
)
_FB2_TEXT_TAGS = {"p", "v", "subtitle", "text-author"}
_FB2_CONTAINER_TAGS = {"cite", "epigraph", "poem", "stanza"}
_HTML_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p"]


@dataclass
class Chapter:
    title: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(paragraph) for paragraph in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    chapters: list[Chapter]


def load_book(path: Path) -> Book:
    loaders = {".txt": _load_txt, ".fb2": _load_fb2, ".epub": _load_epub}
    suffix = path.suffix.lower()
    if suffix not in loaders:
        raise ValueError(
            f"Неподдерживаемый формат «{suffix}». Поддерживаются: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    book = loaders[suffix](path)
    book.chapters = [chapter for chapter in book.chapters if chapter.paragraphs]
    if not book.chapters:
        raise ValueError(f"В файле {path} не найдено текста")
    return book


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


def _load_txt(path: Path) -> Book:
    chapters: list[Chapter] = []
    current: Chapter | None = None
    for raw_line in _read_text(path).splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if len(line) <= 100 and _TXT_HEADING_RE.match(line):
            current = Chapter(title=line)
            chapters.append(current)
            continue
        if current is None:
            current = Chapter(title=path.stem)
            chapters.append(current)
        current.paragraphs.append(line)
    return Book(title=path.stem, author="", chapters=chapters)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((child for child in element if _local(child.tag) == name), None)


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _child_text(element: ET.Element | None, name: str) -> str:
    return _text(_child(element, name))


def _load_fb2(path: Path) -> Book:
    root = ET.parse(path).getroot()
    title, author = path.stem, ""
    title_info = _child(_child(root, "description"), "title-info")
    if title_info is not None:
        title = _child_text(title_info, "book-title") or title
        author_el = _child(title_info, "author")
        names = [_child_text(author_el, name) for name in ("first-name", "middle-name", "last-name")]
        author = " ".join(name for name in names if name)

    chapters: list[Chapter] = []
    for body in _children(root, "body"):
        if body.get("name"):
            continue
        for section in _children(body, "section"):
            _fb2_sections(section, chapters)
    return Book(title=title, author=author, chapters=chapters)


def _fb2_sections(section: ET.Element, chapters: list[Chapter]) -> None:
    nested = _children(section, "section")
    title = _child_text(section, "title")
    if not nested:
        chapters.append(Chapter(title=title, paragraphs=_fb2_paragraphs(section)))
        return
    own_paragraphs = _fb2_paragraphs(section)
    if own_paragraphs:
        chapters.append(Chapter(title=title, paragraphs=own_paragraphs))
    for sub in nested:
        _fb2_sections(sub, chapters)


def _fb2_paragraphs(element: ET.Element) -> list[str]:
    paragraphs: list[str] = []
    for child in element:
        name = _local(child.tag)
        if name in _FB2_TEXT_TAGS:
            text = _text(child)
            if text:
                paragraphs.append(text)
        elif name in _FB2_CONTAINER_TAGS:
            paragraphs.extend(_fb2_paragraphs(child))
    return paragraphs


def _load_epub(path: Path) -> Book:
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if _local(el.tag) == "rootfile")
        opf_path = rootfile.get("full-path", "")
        opf = ET.fromstring(archive.read(opf_path))
        base = posixpath.dirname(opf_path)

        title = _first_text(opf, "title") or path.stem
        author = _first_text(opf, "creator")
        manifest = {el.get("id"): el.get("href") for el in opf.iter() if _local(el.tag) == "item"}
        spine = [el.get("idref") for el in opf.iter() if _local(el.tag) == "itemref"]
        members = set(archive.namelist())

        chapters: list[Chapter] = []
        for idref in spine:
            href = manifest.get(idref)
            if not href:
                continue
            member = posixpath.normpath(posixpath.join(base, unquote(href)))
            if member in members:
                chapters.append(_epub_chapter(archive.read(member)))
    return Book(title=title, author=author, chapters=chapters)


def _first_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if _local(element.tag) == name and element.text and element.text.strip():
            return element.text.strip()
    return ""


def _epub_chapter(data: bytes) -> Chapter:
    soup = BeautifulSoup(data, "html.parser")
    body = soup.body if soup.body is not None else soup
    heading = body.find(["h1", "h2", "h3"])
    title = " ".join(heading.get_text(" ").split()) if heading is not None else ""

    texts = []
    for element in body.find_all(_HTML_BLOCK_TAGS):
        text = " ".join(element.get_text(" ").split())
        if text:
            texts.append(text)
    if not texts:
        texts = [" ".join(line.split()) for line in body.get_text("\n").splitlines()]
        texts = [text for text in texts if text]
    if title and texts and texts[0] == title:
        texts = texts[1:]
    return Chapter(title=title, paragraphs=texts)
