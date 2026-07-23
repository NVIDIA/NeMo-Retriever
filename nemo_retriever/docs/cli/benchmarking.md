# Benchmarking with the Retriever CLI

Retriever has two development benchmarking surfaces:

| Goal | Command | Documentation |
| --- | --- | --- |
| End-to-end ingest, query, and retrieval evaluation | `retriever harness` | [Retriever Harness](../../harness/README.md) |
| Throughput for one internal pipeline stage | `retriever benchmark` | [Stage micro-benchmarks](#stage-micro-benchmarks) |

For product workflows on your own inputs, use `retriever ingest` and
`retriever query` instead.

## End-to-End Benchmarks

The harness owns registered datasets, repeatable runfiles, metric gates, and
stable artifacts. Start with:

```bash
retriever harness list --runsets
retriever harness show jp20_beir
```

Then choose the execution guide:

- [Local and batch library runs](../../harness/LIBRARY.md)
- [Existing-service and managed-Helm runs](../../harness/SERVICE.md)
- [Recurring workstation nightly](../../../ops/retriever-nightly/README.md)

Agentic BEIR evaluation is documented under
[library harness runs](../../harness/LIBRARY.md#evaluate-agentic-retrieval).

## Stage Micro-Benchmarks

`retriever benchmark` measures individual actors rather than an end-to-end
Retriever result. It remains callable for development compatibility but is
hidden from root help.

```bash
retriever benchmark --help
retriever benchmark split --help
retriever benchmark extract --help
retriever benchmark audio-extract --help
retriever benchmark page-elements --help
retriever benchmark ocr --help
retriever benchmark all --help
```

Example:

```bash
retriever benchmark extract ./data/pdf_corpus \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

Stage commands report rows per second, or chunk rows per second for audio. They
do not produce the harness artifact contract and should not be used as retrieval
quality evidence.
