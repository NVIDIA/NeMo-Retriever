from nemo_retriever.evidence.spans import extract_span, split_sentences


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_ignores_blank():
    assert split_sentences("  Only one sentence  ") == ["Only one sentence"]


def test_extract_span_returns_all_when_under_budget():
    text = "The fan costs $150. It is the most expensive."
    assert extract_span(text, "fan cost", max_tokens=50) == text.strip()


def test_extract_span_empty_text():
    assert extract_span("", "anything") == ""


def test_extract_span_bounds_token_count():
    # 300 tokens; a small budget must cap the returned span length.
    text = " ".join(f"w{i}" for i in range(300))
    span = extract_span(text, "w0", max_tokens=40)
    assert len(span.split()) == 40


def test_extract_span_windows_on_query_terms():
    # The relevant region sits in the middle; the window must include it.
    filler = " ".join(["lorem"] * 100)
    text = f"{filler} the premium desk fan costs 150 dollars {filler}"
    span = extract_span(text, "premium desk fan cost dollars", max_tokens=12)
    assert "premium" in span and "150" in span and "dollars" in span
    assert len(span.split()) == 12


def test_extract_span_works_without_sentence_punctuation():
    # Table-like content (numbers, newlines, no periods) must still be bounded.
    text = "\r\n".join(f"row{i} 2024 {i*111}" for i in range(200))
    span = extract_span(text, "2024 555", max_tokens=30)
    assert len(span.split()) == 30
