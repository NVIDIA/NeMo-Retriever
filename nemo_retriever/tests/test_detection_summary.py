from pathlib import Path

import pandas as pd

from nemo_retriever.utils.detection_summary import collect_detection_summary_from_df, print_run_summary


def test_print_run_summary_sorts_evaluation_metrics_by_numeric_k(capsys) -> None:
    print_run_summary(
        processed_pages=10,
        input_path=Path("/tmp/input"),
        hybrid=False,
        lancedb_uri="/tmp/lancedb",
        lancedb_table_name="nv-ingest",
        total_time=10.0,
        ingest_only_total_time=5.0,
        ray_dataset_download_total_time=1.0,
        lancedb_write_total_time=1.0,
        evaluation_total_time=2.0,
        evaluation_metrics={
            "ndcg@10": 0.5,
            "recall@5": 0.4,
            "ndcg@1": 0.6,
            "recall@10": 0.7,
            "ndcg@3": 0.55,
            "recall@1": 0.2,
        },
        evaluation_label="BEIR",
        evaluation_count=20,
    )

    lines = [line.rstrip() for line in capsys.readouterr().out.splitlines()]
    start = lines.index("BEIR metrics:") + 1
    metric_lines = [line.strip() for line in lines[start : start + 6]]

    assert metric_lines == [
        "ndcg@1: 0.6000",
        "ndcg@3: 0.5500",
        "ndcg@10: 0.5000",
        "recall@1: 0.2000",
        "recall@5: 0.4000",
        "recall@10: 0.7000",
    ]


def test_collect_detection_summary_accepts_json_and_python_literal_metadata() -> None:
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/doc_a.pdf",
                "page_number": 0,
                "metadata": (
                    "{'page_elements_v3_num_detections': 3, "
                    "'ocr_chart_detections': 1, "
                    "'page_elements_v3_counts_by_label': {'text': 2}}"
                ),
            },
            {
                "path": "/tmp/doc_a.pdf",
                "page_number": 0,
                "metadata": (
                    '{"page_elements_v3_num_detections": 5, '
                    '"ocr_table_detections": 2, '
                    '"page_elements_v3_counts_by_label": {"text": 4, "table": 1}}'
                ),
            },
        ]
    )

    summary = collect_detection_summary_from_df(df)

    assert summary["pages_seen"] == 1
    assert summary["page_elements_v3_total_detections"] == 5
    assert summary["ocr_chart_total_detections"] == 1
    assert summary["ocr_table_total_detections"] == 2
    assert summary["page_elements_v3_counts_by_label"] == {"table": 1, "text": 4}
