from ttsprodj.text import normalize_text, split_into_chunks


def test_numbers_become_words():
    assert normalize_text("В 1812 году") == "В тысяча восемьсот двенадцать году."
    assert normalize_text("Цена 3,5 рубля") == "Цена три целых пять десятых рубля."


def test_grouped_thousands_are_joined():
    result = normalize_text("Продано 1 000 книг")
    assert "1000" not in result and "000" not in result
    assert "тысяч" in result


def test_percent_and_symbols():
    assert normalize_text("Скидка 5%") == "Скидка пять процентов."


def test_dialogue_dashes_are_removed():
    assert normalize_text("— Привет, — сказал он.") == "Привет, сказал он."


def test_quotes_are_removed():
    assert normalize_text("Он сказал «привет».") == "Он сказал привет."


def test_paragraph_gets_final_punctuation():
    assert normalize_text("Без точки") == "Без точки."
    assert normalize_text("Конец абзаца,") == "Конец абзаца."


def test_chunks_respect_limit_and_mark_paragraph_ends():
    sentence = "Это предложение средней длины для проверки разбиения."
    paragraphs = [" ".join([sentence] * 6), "Короткий абзац."]
    chunks = split_into_chunks(paragraphs, max_chars=120)

    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert chunks[-1].text == "Короткий абзац."
    assert chunks[-1].end_of_paragraph is True
    first_paragraph = [chunk for chunk in chunks if not chunk.end_of_paragraph or chunk is not chunks[-1]]
    assert sum(chunk.end_of_paragraph for chunk in first_paragraph) == 1


def test_empty_paragraphs_are_skipped():
    assert split_into_chunks(["", "   ", "—"]) == []
