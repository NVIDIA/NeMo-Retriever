from pathlib import Path

from nemo_retriever.recall.beir import (
    BeirConfig,
    BeirDataset,
    build_beir_run_from_hits,
    build_qrels_by_query_id,
    build_queries_by_id,
    compute_beir_metrics,
    evaluate_lancedb_beir,
    load_beir_dataset,
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


def test_load_beir_dataset_supports_bo767_csv_pdf_page_modality(tmp_path: Path) -> None:
    annotations = tmp_path / "bo767_annotations.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("bo767_csv", dataset_name=str(annotations), doc_id_field="pdf_page_modality")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1_text": 1},
        "1": {"1002_5_table": 1},
    }


def test_load_beir_dataset_supports_bo767_csv_pdf_page(tmp_path: Path) -> None:
    annotations = tmp_path / "bo767_annotations.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("bo767_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1": 1},
        "1": {"1002_5": 1},
    }


def test_load_beir_dataset_supports_bo10k_csv_pdf_page_modality(tmp_path: Path) -> None:
    annotations = tmp_path / "digital_corpora_10k_annotations.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("bo10k_csv", dataset_name=str(annotations), doc_id_field="pdf_page_modality")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1_text": 1},
        "1": {"1002_5_table": 1},
    }


def test_load_beir_dataset_supports_earnings_csv_pdf_page(tmp_path: Path) -> None:
    annotations = tmp_path / "earnings_consulting_multimodal.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
                "chart,What is doc c?,Answer C,1003.pdf,2",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("earnings_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.query_ids == ["0", "1", "2"]
    assert dataset.queries == ["What is doc a?", "What is doc b?", "What is doc c?"]
    assert dataset.qrels == {
        "0": {"1001_1": 1},
        "1": {"1002_5": 1},
        "2": {"1003_3": 1},
    }


def test_load_beir_dataset_supports_jp20_csv_pdf_page(tmp_path: Path) -> None:
    annotations = tmp_path / "jp20_query_gt.csv"
    annotations.write_text(
        "\n".join(
            [
                "query,pdf,page,pdf_page",
                "What is doc a?,1001,0,1001_1",
                "What is doc b?,1002.pdf,4,1002_5",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("jp20_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1": 1},
        "1": {"1002_5": 1},
    }


def test_load_beir_dataset_supports_financebench_json_pdf_basename(tmp_path: Path) -> None:
    annotations = tmp_path / "financebench_train.json"
    annotations.write_text(
        '[{"id":"q1","question":"What is revenue?","contexts":[{"filename":"AAPL_2023.pdf"}]},'
        '{"id":"q2","question":" What is margin? ","contexts":[{"filename":"MSFT_2022"}]}]',
        encoding="utf-8",
    )

    dataset = load_beir_dataset("financebench_json", dataset_name=str(annotations), doc_id_field="pdf_basename")

    assert dataset.query_ids == ["q1", "q2"]
    assert dataset.queries == ["What is revenue?", " What is margin? "]
    assert dataset.qrels == {
        "q1": {"AAPL_2023": 1},
        "q2": {"MSFT_2022": 1},
    }


def test_load_beir_dataset_preserves_dotted_pdf_basenames_without_extension(tmp_path: Path) -> None:
    annotations = tmp_path / "earnings_consulting_multimodal.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is fair value?,Answer A,3.-Facebook-Reports-Third-Quarter-2016-Results,0",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("earnings_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.qrels == {"0": {"3.-Facebook-Reports-Third-Quarter-2016-Results_1": 1}}


def test_load_beir_dataset_preserves_query_whitespace_for_retrieval_parity(tmp_path: Path) -> None:
    annotations = tmp_path / "earnings_consulting_multimodal.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is the Apple arcade? ,Answer A,1001,0",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("earnings_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.queries == ["What is the Apple arcade? "]


def test_build_beir_run_from_hits_synthesizes_pdf_page_modality() -> None:
    raw_hits = [
        [
            {
                "source": '{"source_id": "/tmp/doc_a.pdf"}',
                "page_number": 7,
                "metadata": "{'_content_type': 'table_caption'}",
            },
            {
                "source_id": "/tmp/doc_a.pdf",
                "page_number": 7,
                "metadata": '{"content_metadata": {"type": "table"}}',
            },
            {
                "source_id": "/tmp/doc_b.pdf",
                "page_number": 3,
                "metadata": "{'content_metadata': {'type': 'text'}}",
            },
        ]
    ]

    run = build_beir_run_from_hits(["q1"], raw_hits, doc_id_field="pdf_page_modality")

    assert list(run["q1"].keys()) == ["doc_a_7_table", "doc_b_3_text"]
    assert run["q1"]["doc_a_7_table"] > run["q1"]["doc_b_3_text"]


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
            expected_kwargs = {
                "vdb": "lancedb",
                "vdb_kwargs": {
                    "uri": "/tmp/lancedb",
                    "table_name": "nv-ingest",
                    "hybrid": False,
                    "nprobes": 0,
                    "refine_factor": 10,
                },
                "embedder": "embedder",
                "embedding_endpoint": "http://embed.example/v1",
                "embedding_api_key": "secret",
                "embedding_use_grpc": False,
                "top_k": 10,
                "local_hf_device": None,
                "local_hf_cache_dir": None,
                "local_hf_batch_size": 32,
                "local_query_embed_backend": "hf",
                "reranker": False,
                "reranker_model_name": "nvidia/llama-nemotron-rerank-1b-v2",
                "reranker_endpoint": None,
                "reranker_api_key": "",
                "reranker_batch_size": 32,
                "local_reranker_backend": "vllm",
            }
            missing_keys = set(expected_kwargs) - set(kwargs)
            assert not missing_keys
            assert {key: kwargs[key] for key in expected_kwargs} == expected_kwargs
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
        embedding_http_endpoint="http://embed.example/v1",
        embedding_api_key=" secret ",
        loader="vidore_hf",
        dataset_name="vidore_v3_computer_science",
    )

    loaded_dataset, _raw_hits, _run, metrics = evaluate_lancedb_beir(cfg)

    assert loaded_dataset == dataset
    assert metrics["ndcg@10"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert "embed_use_vllm" not in retriever_instances[0].kwargs
    assert retriever_instances[0].kwargs.get("local_query_embed_backend") == "hf"
    assert retriever_instances[0].kwargs.get("local_reranker_backend") == "vllm"
