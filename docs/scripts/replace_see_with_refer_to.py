#!/usr/bin/env python3
"""Replace instructional 'see' with 'refer to' in docs/docs/extraction/*.md."""
from __future__ import annotations

from pathlib import Path

EXTRACTION = Path(__file__).resolve().parents[1] / "docs" / "extraction"

# Keep these phrases verbatim (do not replace inner 'see').
KEEP = [
    "You should see",
    "you should see",
    "might see errors",
    "might see an error",
    "might see log messages",
    "For example, you might",
    "When you run a job you might",
    "In rare cases, when you run a job you might",
    "the embedding service might fail to start and you might see",
]


def protect(s: str) -> tuple[str, dict[str, str]]:
    tokens: dict[str, str] = {}
    out = s
    for i, phrase in enumerate(KEEP):
        key = f"\0KEEP{i}\0"
        if phrase in out:
            out = out.replace(phrase, key)
            tokens[key] = phrase
    return out, tokens


def restore(s: str, tokens: dict[str, str]) -> str:
    for k, v in tokens.items():
        s = s.replace(k, v)
    return s


def transform(s: str) -> str:
    s, tok = protect(s)
    # Longer patterns first
    reps: list[tuple[str, str]] = [
        ("For more information, see the following pages:", "For more information, refer to the following pages:"),
        ("For more information, see the", "For more information, refer to the"),
        ("For more information, see ", "For more information, refer to "),
        ("For code examples, see the", "For code examples, refer to the"),
        ("For detailed metadata schema documentation, see:**", "For detailed metadata schema documentation, refer to:**"),
        ("For the complete and up-to-date list of pipeline stages, see the", "For the complete and up-to-date list of pipeline stages, refer to the"),
        ("For advanced usage patterns, see the existing", "For advanced usage patterns, refer to the existing"),
        ("For scheduling and GPU partitioning, see the", "For scheduling and GPU partitioning, refer to the"),
        ("For details, see ", "For details, refer to "),
        ("For a concise comparison, see ", "For a concise comparison, refer to "),
        ("For harnesses and metrics, see ", "For harnesses and metrics, refer to "),
        (
            "For guidance on choosing between static and dynamic scaling modes, and how to configure them in `docker-compose.yaml`, see ",
            "For guidance on choosing between static and dynamic scaling modes, and how to configure them in `docker-compose.yaml`, refer to ",
        ),
        ("For more, see ", "For more, refer to "),
        ("deeper topics, see ", "deeper topics, refer to "),
        ("For operations topics, see ", "For operations topics, refer to "),
        (
            "Documentation here focuses on stores used in the library and harnesses, such as LanceDB and Milvus, and cuVS where it applies. See ",
            "Documentation here focuses on stores used in the library and harnesses, such as LanceDB and Milvus, and cuVS where it applies. Refer to ",
        ),
        (
            "For NVIDIA AI Blueprint links, solution cards, enterprise RAG resources, and related product landing pages, see ",
            "For NVIDIA AI Blueprint links, solution cards, enterprise RAG resources, and related product landing pages, refer to ",
        ),
        ("For the audio and speech path, see ", "For the audio and speech path, refer to "),
        (
            "For visual text and OCR, scanned or image-heavy content often uses OCR-oriented extract methods. See ",
            "For visual text and OCR, scanned or image-heavy content often uses OCR-oriented extract methods. Refer to ",
        ),
        ("For end-to-end RAG stacks that include multimodal ingestion, see ", "For end-to-end RAG stacks that include multimodal ingestion, refer to "),
        ("3. **Tune extraction for your content.** See ", "3. **Tune extraction for your content.** Refer to "),
        ("1. **Query.** Run searches against your vector store with filters as needed. See ", "1. **Query.** Run searches against your vector store with filters as needed. Refer to "),
        ("3. **Rerank.** Apply a reranker NIM for a second-stage score on candidates. See ", "3. **Rerank.** Apply a reranker NIM for a second-stage score on candidates. Refer to "),
        ("For more information about other profiles, see ", "For more information about other profiles, refer to "),
        ("Python 3.12 or later is required (see [Prerequisites]", "Python 3.12 or later is required (refer to [Prerequisites]"),
        ("(see [Return type]", "(refer to [Return type]"),
        ("(see below)", "(refer to below)"),
        ("(see `recall_utils.py`)", "(refer to `recall_utils.py`)"),
        ("- **Configuration**: See `config.py`", "- **Configuration**: Refer to `config.py`"),
        ("- **Test utilities**: See `interact.py`", "- **Test utilities**: Refer to `interact.py`"),
        ("- **Docker setup**: See project root README", "- **Docker setup**: Refer to project root README"),
        ("- **API documentation**: See `docs/`", "- **API documentation**: Refer to `docs/`"),
        ("# Create a custom model interface (see examples below)", "# Create a custom model interface (refer to examples below)"),
        ("(see all duplicate messages)", "(refer to all duplicate messages)"),
        ("## See also", "## Related topics"),
        ("See [**NimClient", "Refer to [**NimClient"),
        ("See the [Profile Information]", "Refer to the [Profile Information]"),
        ("See the [", "Refer to the ["),
        ("See [", "Refer to ["),
        (", see [", ", refer to ["),
        ("; see [", "; refer to ["),
        (": see [", ": refer to ["),
        (". See [", ". Refer to ["),
        ("(see the [", "(refer to the ["),
        ("(see [", "(refer to ["),
        ("; see the [", "; refer to the ["),
        ("; see [", "; refer to ["),
        (" see [", " refer to ["),
        ("see the [", "refer to the ["),
    ]
    for old, new in reps:
        s = s.replace(old, new)
    return restore(s, tok)


def main() -> None:
    for path in sorted(EXTRACTION.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        updated = transform(raw)
        if raw != updated:
            path.write_text(updated, encoding="utf-8", newline="\n")
            print(path.name)


if __name__ == "__main__":
    main()
