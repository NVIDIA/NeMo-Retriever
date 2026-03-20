from typer.testing import CliRunner

import nemo_retriever.examples.batch_pipeline as batch_pipeline
from nemo_retriever.utils.input_files import resolve_input_patterns

RUNNER = CliRunner()


class _FakeDataset:
    def materialize(self):
        return self

    def take_all(self):
        return []

    def groupby(self, _key):
        class _FakeGrouped:
            @staticmethod
            def count():
                class _FakeCounted:
                    @staticmethod
                    def count():
                        return 1

                return _FakeCounted()

        return _FakeGrouped()


class _FakeIngestResult:
    def get_dataset(self):
        return _FakeDataset()


class _FakeErrorRows:
    def materialize(self):
        return self

    def count(self) -> int:
        return 0


class _FakeIngestor:
    def __init__(self) -> None:
        self.extract_params = None
        self.embed_params = None
        self.file_patterns = None

    def files(self, file_patterns):
        self.file_patterns = file_patterns
        return self

    def extract(self, params):
        self.extract_params = params
        return self

    def extract_image_files(self, params):
        self.extract_params = params
        return self

    def extract_txt(self, params):
        return self

    def extract_html(self, params):
        return self

    def split(self, params):
        return self

    def embed(self, params):
        self.embed_params = params
        return self

    def ingest(self, params=None):
        return _FakeIngestResult()

    def get_error_rows(self, dataset=None):
        return _FakeErrorRows()


def test_resolve_input_file_patterns_recurses_for_directory_inputs(tmp_path) -> None:
    dataset_dir = tmp_path / "earnings_consulting"
    dataset_dir.mkdir()

    pdf_patterns = resolve_input_patterns(dataset_dir, "pdf")
    txt_patterns = resolve_input_patterns(dataset_dir, "txt")
    doc_patterns = resolve_input_patterns(dataset_dir, "doc")

    assert pdf_patterns == [str(dataset_dir / "**" / "*.pdf")]
    assert txt_patterns == [str(dataset_dir / "**" / "*.txt")]
    assert doc_patterns == [str(dataset_dir / "**" / "*.docx"), str(dataset_dir / "**" / "*.pptx")]


def test_batch_pipeline_accepts_multimodal_embed_and_page_image_flags(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")
    missing_query_csv = tmp_path / "missing.csv"

    fake_ingestor = _FakeIngestor()
    monkeypatch.setattr(batch_pipeline, "create_ingestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(batch_pipeline, "_ensure_lancedb_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "handle_lancedb", lambda *args, **kwargs: None)

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--query-csv",
            str(missing_query_csv),
            "--embed-modality",
            "text_image",
            "--embed-granularity",
            "page",
            "--extract-infographics",
            "--no-extract-page-as-image",
        ],
    )

    assert result.exit_code == 0
    assert isinstance(fake_ingestor.file_patterns, list)
    assert fake_ingestor.extract_params.extract_infographics is True
    assert fake_ingestor.extract_params.extract_page_as_image is False
    assert fake_ingestor.embed_params.embed_modality == "text_image"
    assert fake_ingestor.embed_params.embed_granularity == "page"


def test_batch_pipeline_routes_beir_mode_to_evaluator(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "sample.pdf").write_text("placeholder", encoding="utf-8")

    fake_ingestor = _FakeIngestor()
    monkeypatch.setattr(batch_pipeline, "create_ingestor", lambda *args, **kwargs: fake_ingestor)
    monkeypatch.setattr(batch_pipeline, "_ensure_lancedb_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "handle_lancedb", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_pipeline, "print_run_summary", lambda *args, **kwargs: None)

    class _FakeTable:
        def count_rows(self) -> int:
            return 1

    class _FakeDb:
        def open_table(self, _name):
            return _FakeTable()

    class _FakeLanceModule:
        @staticmethod
        def connect(_uri):
            return _FakeDb()

    monkeypatch.setattr(batch_pipeline, "_lancedb", lambda: _FakeLanceModule())

    captured = {}

    def _fake_evaluate(cfg):
        captured["cfg"] = cfg
        return type("Dataset", (), {"query_ids": ["1", "2"]})(), [], {}, {"ndcg@10": 0.75, "recall@5": 0.6}

    monkeypatch.setattr(batch_pipeline, "evaluate_lancedb_beir", _fake_evaluate)

    result = RUNNER.invoke(
        batch_pipeline.app,
        [
            str(dataset_dir),
            "--evaluation-mode",
            "beir",
            "--beir-loader",
            "vidore_hf",
            "--beir-dataset-name",
            "vidore_v3_computer_science",
            "--beir-k",
            "5",
            "--beir-k",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert captured["cfg"].loader == "vidore_hf"
    assert captured["cfg"].dataset_name == "vidore_v3_computer_science"
    assert tuple(captured["cfg"].ks) == (5, 10)
