from pathlib import Path

from nemo_retriever.utils.detection_summary import print_run_summary


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
