from nemo_retriever.pdf.split import PDFSplitActor
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.graph import FileListLoaderOperator
from nemo_retriever.ocr.ocr import OCRActor
from nemo_retriever.params import EmbedParams
from nemo_retriever.operators.content import ExplodeContentActor
from nemo_retriever.operators.embedding import _BatchEmbedActor
from typing import Any
import glob

extract_kwargs: dict[str, Any] = {
    "method": "pdfium",
    "dpi": 300,
    "extract_text": True,
    "extract_tables": True,
    "extract_charts": True,
    "extract_page_as_image": True,
}

detect_kwargs: dict[str, Any] = {}

ocr_kwargs: dict[str, Any] = {
    "extract_text": True,
    "extract_tables": True,
    "extract_charts": True,
}

embed_params = EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2")

# -- Build the graph using >> chaining -------------------------------------
explode_kwargs: dict[str, Any] = {"modality": "text"}

graph = (
    FileListLoaderOperator()  # Start with loading files from a folder
    >> PDFSplitActor()
    >> PDFExtractionActor(**extract_kwargs)
    >> PageElementDetectionActor(**detect_kwargs)
    >> OCRActor(**ocr_kwargs)
    >> ExplodeContentActor(**explode_kwargs)
    >> _BatchEmbedActor(params=embed_params)

)
file_patterns = glob.glob(str("/raid/nv-ingest/data/*.pdf"))
results = graph.execute(file_patterns)  # Replace with actual folder path containing mixed files
assert results[0].shape[0] >= len(file_patterns) * 2, "Expected at least 2 rows per PDF (text and image)"
assert all(results[0]["text_embeddings_1b_v2_has_embedding"]) == True
