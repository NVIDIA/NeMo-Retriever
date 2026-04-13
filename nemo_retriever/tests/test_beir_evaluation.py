from nemo_retriever.recall.beir import (
    BeirConfig,
    BeirDataset,
    build_beir_run_from_hits,
    build_qrels_by_query_id,
    build_queries_by_id,
    compute_beir_metrics,
    evaluate_lancedb_beir,
)


def test_build_queries_by_id_filters_language() -> None:
    rows = [
        {"query_id": 1, "query": "what is a qubit?", "language": "english"},
        {"query_id": 2, "query": "bonjour", "language": "french"},
    ]

    query_ids, queries = build_queries_by_id(rows, query_language="english")

    assert query_ids == ["1"]
    assert queries == ["what is a qubit?"]


def test_build_qrels_by_query_id_formats_nested_dict() -> None:
    rows = [
        {"query_id": 1, "corpus_id": "doc_a", "score": 1},
        {"query_id": 1, "corpus_id": "doc_b", "score": 2},
        {"query_id": 2, "corpus_id": "doc_c", "score": 1},
    ]

    qrels = build_qrels_by_query_id(rows, allowed_query_ids={"1"})

    assert qrels == {"1": {"doc_a": 1, "doc_b": 2}}


def test_build_beir_run_from_hits_uses_pdf_basename_and_dedupes() -> None:
    raw_hits = [
        [
            {"pdf_basename": "doc_a", "source_id": "/tmp/doc_a.pdf"},
            {"pdf_basename": "doc_a", "source_id": "/tmp/doc_a.pdf"},
            {"pdf_basename": "doc_b", "source_id": "/tmp/doc_b.pdf"},
        ]
    ]

    run = build_beir_run_from_hits(["q1"], raw_hits, doc_id_field="pdf_basename")

    assert list(run["q1"].keys()) == ["doc_a", "doc_b"]
    assert run["q1"]["doc_a"] > run["q1"]["doc_b"]


def test_compute_beir_metrics_returns_expected_cutoffs() -> None:
    qrels = {
        "q1": {"doc_a": 1},
        "q2": {"doc_b": 1},
    }
    run = {
        "q1": {"doc_a": 2.0, "doc_c": 1.0},
        "q2": {"doc_c": 2.0, "doc_b": 1.0},
    }

    metrics = compute_beir_metrics(qrels, run, ks=(1, 2))

    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["ndcg@1"] == 0.5


def test_evaluate_lancedb_beir_uses_loader_and_retriever(monkeypatch) -> None:
    dataset = BeirDataset(
        dataset_name="vidore_v3_computer_science",
        query_ids=["1"],
        queries=["what is a qubit?"],
        qrels={"1": {"doc_a": 1}},
    )

    monkeypatch.setattr(
        "nemo_retriever.recall.beir.load_beir_dataset",
        lambda *args, **kwargs: dataset,
    )

    retriever_instances: list = []

    class _FakeRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            retriever_instances.append(self)

        def queries(self, queries):
            assert queries == ["what is a qubit?"]
            return [[{"pdf_basename": "doc_a", "source_id": "/tmp/doc_a.pdf"}]]

    monkeypatch.setattr("nemo_retriever.recall.beir.Retriever", _FakeRetriever)

    cfg = BeirConfig(
        lancedb_uri="/tmp/lancedb",
        lancedb_table="nv-ingest",
        embedding_model="embedder",
        loader="vidore_hf",
        dataset_name="vidore_v3_computer_science",
    )

    loaded_dataset, _raw_hits, _run, metrics = evaluate_lancedb_beir(cfg)

    assert loaded_dataset == dataset
    assert metrics["ndcg@10"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert "embed_use_vllm" not in retriever_instances[0].kwargs
    assert retriever_instances[0].kwargs.get("local_query_embed_backend") == "auto"
