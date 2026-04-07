from nemo_retriever.harness.parsers import StreamMetrics, parse_stream_text


def test_parse_stream_text_extracts_done_recall_and_pages_per_second() -> None:
    stdout = """
[done] 20 files, 3181 pages in 122.3s
Recall metrics (matching nemo_retriever.recall.core):
  recall@1: 0.6087
  recall@5: 0.9043
  recall@10: 0.9565
Pages/sec (ingest only; excludes Ray startup and recall): 15.77
"""
    metrics = parse_stream_text(stdout)
    assert metrics.files == 20
    assert metrics.pages == 3181
    assert metrics.ingest_secs == 122.3
    assert metrics.pages_per_sec_ingest == 15.77
    assert metrics.recall_metrics == {
        "recall@1": 0.6087,
        "recall@5": 0.9043,
        "recall@10": 0.9565,
    }
    assert metrics.rows_processed is None
    assert metrics.rows_per_sec_ingest is None


def test_parse_stream_text_extracts_rows_and_logger_prefixed_recall_lines() -> None:
    stdout = """
2026-03-06 20:12:42,120 INFO nemo_retriever.examples.batch_pipeline: \
Ingestion complete. 3181 rows procesed in 151.40 seconds. 21.01 PPS
2026-03-06 20:12:42,121 INFO nemo_retriever.examples.batch_pipeline: \
Recall metrics (matching nemo_retriever.recall.core):
2026-03-06 20:12:42,122 INFO nemo_retriever.examples.batch_pipeline:   recall@1: 0.6087
2026-03-06 20:12:42,123 INFO nemo_retriever.examples.batch_pipeline:   recall@5: 0.9043
2026-03-06 20:12:42,124 INFO nemo_retriever.examples.batch_pipeline:   recall@10: 0.9565
"""
    metrics = parse_stream_text(stdout)
    assert metrics.files is None
    assert metrics.pages is None
    assert metrics.ingest_secs == 151.40
    assert metrics.pages_per_sec_ingest is None
    assert metrics.rows_processed == 3181
    assert metrics.rows_per_sec_ingest == 21.01
    assert metrics.recall_metrics == {
        "recall@1": 0.6087,
        "recall@5": 0.9043,
        "recall@10": 0.9565,
    }


def test_stream_metrics_handles_non_recall_lines_after_recall_block() -> None:
    metrics = StreamMetrics()
    metrics.consume("Recall metrics (matching nemo_retriever.recall.core):\n")
    metrics.consume("  recall@5: 0.9043\n")
    metrics.consume("Pages processed: 1933\n")
    metrics.consume("  recall@10: 0.9565\n")
    assert metrics.recall_metrics == {"recall@5": 0.9043}


def test_parse_stream_text_extracts_beir_metrics_block() -> None:
    stdout = """
BEIR metrics:
  ndcg@10: 0.7421
  recall@5: 0.6234
"""
    metrics = parse_stream_text(stdout)
    assert metrics.recall_metrics == {}
    assert metrics.evaluation_metrics == {
        "ndcg@10": 0.7421,
        "recall@5": 0.6234,
    }


def test_parse_stream_text_extracts_ingest_metrics_from_run_summary() -> None:
    stdout = """
===== Run Summary - 2026-03-24 12:48:30 UTC =====
Run Configuration:
\tInput path: /tmp/vidore_v3_hr
\tHybrid: False
Runtimes:
\tTotal pages processed: 1110 from /tmp/vidore_v3_hr
\tIngestion only time: 152.56s / 0:02:32.562
\tRay dataset download time: 3.07s / 0:00:03.075
\tLanceDB Write Time: 0.32s / 0:00:00.317
\tBEIR time: 21.47s / 0:00:21.470
PPS:
\tIngestion only PPS: 7.28
\tIngestion + LanceDB Write PPS: 7.26
\tBEIR QPS: 88.87
\tTotal - Processed: 1110 pages in 181.89s / 0:03:01.889 @ 6.10 PPS
BEIR metrics:
  ndcg@1: 0.4966
  ndcg@3: 0.4907
  ndcg@5: 0.5008
  ndcg@10: 0.5284
  recall@1: 0.2027
  recall@3: 0.3607
  recall@5: 0.4491
  recall@10: 0.5747
"""
    metrics = parse_stream_text(stdout)
    assert metrics.ingest_secs == 152.56
    assert metrics.pages_per_sec_ingest == 7.28
    assert metrics.evaluation_metrics["ndcg@10"] == 0.5284
    assert metrics.evaluation_metrics["recall@5"] == 0.4491
