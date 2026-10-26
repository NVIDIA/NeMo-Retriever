# Benchmarking with the Retriever CLI

NVIDIA runs end-to-end Retriever benchmarks with an internal benchmarking
harness. To evaluate retrieval on your own setup, use the example notebooks in
[`examples/`](../../../examples/), such as
[ViDoRe v3 HR ingestion and retrieval evaluation](../../../examples/vidore_v3_hr.ipynb)
and [RAG evaluation with RAGAS](../../../examples/nrl_ragas.ipynb).
The product CLI retains internal stage micro-benchmarks for focused development
measurements. For product workflows on your own inputs, use
`retriever ingest` and `retriever query`.

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

The following example extracts from a directory of PDF files that you supply.

```bash
retriever benchmark extract /path/to/your/pdfs \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

Stage commands report rows per second, or chunk rows per second for audio. They
do not produce the NRB artifact contract and should not be used as retrieval
quality evidence.
