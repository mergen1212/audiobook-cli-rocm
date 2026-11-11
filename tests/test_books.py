import zipfile
from pathlib import Path

import pytest

from ttsprodj.books import load_book


def test_txt_splits_chapters_by_headings(tmp_path: Path):
    path = tmp_path / "kniga.txt"
    path.write_text(
        "Предисловие к тексту.\n\nГлава 1. Начало\nПервый абзац.\nВторой абзац.\n\nГлава 2\nТретий абзац.\n",
        encoding="utf-8",
    )
    book = load_book(path)

    assert [chapter.title for chapter in book.chapters] == ["kniga", "Глава 1. Начало", "Глава 2"]
    assert book.chapters[1].paragraphs == ["Первый абзац.", "Второй абзац."]
    assert book.chapters[2].paragraphs == ["Третий абзац."]


def test_txt_cp1251(tmp_path: Path):
    path = tmp_path / "old.txt"
    path.write_bytes("Привет, мир.".encode("cp1251"))
    book = load_book(path)
    assert book.chapters[0].paragraphs == ["Привет, мир."]


def test_fb2_nested_sections_become_chapters(tmp_path: Path):
    path = tmp_path / "book.fb2"
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <author><first-name>Иван</first-name><last-name>Петров</last-name></author>
      <book-title>Тестовая книга</book-title>
    </title-info>
  </description>
  <body>
    <section>
      <title><p>Глава первая</p></title>
      <p>Текст первой главы.</p>
    </section>
    <section>
      <title><p>Глава вторая</p></title>
      <section><title><p>Часть А</p></title><p>Текст части А.</p></section>
      <section><title><p>Часть Б</p></title><p>Текст части Б.</p></section>
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )
    book = load_book(path)

    assert book.title == "Тестовая книга"
    assert book.author == "Иван Петров"
    assert [chapter.title for chapter in book.chapters] == ["Глава первая", "Часть А", "Часть Б"]
    assert book.chapters[0].paragraphs == ["Текст первой главы."]


def _make_epub(path: Path) -> None:
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Эпаб</dc:title>
    <dc:creator>Анна Смирнова</dc:creator>
  </metadata>
  <manifest>
    <item id="c1" href="text/one%20two.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="two.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>"""
    chapter_one = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title></head>
<body><h1>Первая глава</h1><p>Абзац один.</p><p>Абзац два.</p></body></html>"""
    chapter_two = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Вторая глава.</p></body></html>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/text/one two.xhtml", chapter_one)
        archive.writestr("OEBPS/two.xhtml", chapter_two)


def test_epub_follows_spine(tmp_path: Path):
    path = tmp_path / "book.epub"
    _make_epub(path)
    book = load_book(path)

    assert book.title == "Эпаб"
    assert book.author == "Анна Смирнова"
    assert [chapter.title for chapter in book.chapters] == ["Первая глава", ""]
    assert book.chapters[0].paragraphs == ["Абзац один.", "Абзац два."]
    assert book.chapters[1].paragraphs == ["Вторая глава."]


def test_unsupported_format(tmp_path: Path):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        load_book(path)
