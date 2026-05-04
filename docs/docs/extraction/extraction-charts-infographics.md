# Charts and infographics

Charts are detected as page-level bounding boxes by the **yolox-page-elements-v3** model. NeMo Retriever crops each chart and runs OCR over the crop; the OCR text is emitted verbatim into the chart metadata, with no rearrangement and no `-` separators between detected regions. Outputs use the same metadata schema as other extracted objects.

**Related**

- [What is NeMo Retriever Library?](overview.md)
- [Support matrix](support-matrix.md)
- [Multimodal embeddings (VLM)](embedding.md) when you treat graphics as images for embedding
