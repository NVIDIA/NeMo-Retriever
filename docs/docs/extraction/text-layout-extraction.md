# Text and layout extraction

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.


For PDFs, NeMo Retriever Library typically uses **pdfium**-based extraction with configurable depth and paths. Scanned or mixed pages may use hybrid or OCR-oriented methods. For `extract_method` options such as `pdfium`, `pdfium_hybrid`, and `ocr`, refer to the [Python API reference](python-api-reference.md).

**Related**

- [What is NeMo Retriever Library?](overview.md)
- [OCR and scanned documents](extraction-ocr-scanned.md)
- [Chunking and splitting](chunking.md)
