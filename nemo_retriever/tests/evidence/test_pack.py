from nemo_retriever.evidence import build_evidence_pack


def _hit(text, *, source_id, page, content_type, score, needs_ocr=False):
    return {
        "text": text,
        "source_id": source_id,
        "page_number": page,
        "content_type": content_type,
        "_score": score,
        "metadata": {"needs_ocr_for_text": needs_ocr, "content_metadata": {}},
    }


def test_build_pack_shapes_items_and_coverage():
    hits = [
        _hit("The premium desk fan costs 150 dollars. Shipping extra.",
             source_id="multimodal_test", page=1, content_type="text", score=5.0),
        _hit("Bar chart of gadget prices.",
             source_id="multimodal_test", page=1, content_type="chart", score=1.0),
    ]
    pack = build_evidence_pack(hits, "how much does the desk fan cost", max_tokens=120)

    assert len(pack["evidence"]) == 2
    top = pack["evidence"][0]
    # text is under the token budget -> returned whole
    assert top["span"] == "The premium desk fan costs 150 dollars. Shipping extra."
    assert top["fidelity"] == "verbatim"          # text + not needs_ocr
    assert top["modality"] == "text"
    assert top["citation"] == {"source": "multimodal_test", "locator": "p.1"}
    assert top["handle"] == "multimodal_test|1|0"
    assert top["confidence"] == 1.0               # highest score -> norm 1.0; both same page -> corrob 1.0

    chart = pack["evidence"][1]
    assert chart["fidelity"] == "ocr"             # chart -> ocr per _derive_fidelity
    assert pack["coverage"]["n_hits"] == 2


def test_build_pack_reads_modality_fidelity_from_metadata():
    # Real query_documents hits carry modality/fidelity in metadata (type/fidelity),
    # not a top-level content_type, and source as a full path.
    hits = [{
        "text": "Net revenue was 12,345 million in 2024.",
        "source_id": "/abs/path/jpmorgan_chase_2024.pdf",
        "page_number": 296,
        "_score": 3.0,
        "metadata": {"type": "text", "fidelity": "verbatim", "content_metadata": {}},
    }]
    item = build_evidence_pack(hits, "net revenue 2024")["evidence"][0]
    assert item["modality"] == "text"
    assert item["fidelity"] == "verbatim"
    assert item["citation"] == {"source": "jpmorgan_chase_2024", "locator": "p.296"}
    assert item["handle"] == "jpmorgan_chase_2024|296|0"


def test_build_pack_empty_hits():
    pack = build_evidence_pack([], "anything")
    assert pack["evidence"] == []
    assert pack["coverage"]["weak"] is True
