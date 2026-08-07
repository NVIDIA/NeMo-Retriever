# Benchmarking with the Retriever CLI

End-to-end experiments are maintained in the NeMo Retriever Benchmark (NRB)
repository. The product CLI retains internal stage micro-benchmarks for focused
development measurements. For product workflows on your own inputs, use
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

Example:

```bash
retriever benchmark extract ./data/pdf_corpus \
  --pdf-extract-batch-size 8 \
  --pdf-extract-actors 4
```

Stage commands report rows per second, or chunk rows per second for audio. They
do not produce the NRB artifact contract and should not be used as retrieval
quality evidence.
