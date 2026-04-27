# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path

MODEL_ID = os.environ.get("PARAKEET_CTC_MODEL_ID", "nvidia/parakeet-ctc-1.1b")
DEFAULT_AUDIO_PATH = "/datasets/nv-ingest/audio_retrieval_data_mp3/How_to_Convert_a_Word_Document_to_PDF_Dropbox.mp3"
AUDIO_PATH = Path(os.environ.get("PARAKEET_CTC_TEST_AUDIO", DEFAULT_AUDIO_PATH))
DEFAULT_EXPECTED_PHRASES = (
    "word document",
    "convert to a pdf",
    "free converter tool from dropbox",
)
EXPECTED_PHRASES = tuple(
    phrase.strip()
    for phrase in os.environ.get("PARAKEET_CTC_EXPECTED_PHRASES", "||".join(DEFAULT_EXPECTED_PHRASES)).split("||")
    if phrase.strip()
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_PARAKEET_CTC_HF_TEST") != "1",
        reason="Set RUN_PARAKEET_CTC_HF_TEST=1 to run this large-model compatibility test.",
    ),
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def test_parakeet_ctc_hf_transcribes_single_audio_file() -> None:
    """Smoke-test the repo-like local Parakeet CTC Transformers path."""
    librosa = pytest.importorskip("librosa")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    AutoModelForCTC = transformers.AutoModelForCTC
    AutoProcessor = transformers.AutoProcessor
    if not AUDIO_PATH.exists():
        pytest.skip(f"Test audio file does not exist: {AUDIO_PATH}")
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map=device_map,
    )
    audio, _ = librosa.load(str(AUDIO_PATH), sr=16000, mono=True)
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    # Matches nemo_retriever's local Parakeet path.
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    transcript = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(
        {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "audio_path": str(AUDIO_PATH),
            "transcript_preview": transcript[:300],
        }
    )
    normalized_transcript = _normalize_text(transcript)
    missing = [phrase for phrase in EXPECTED_PHRASES if _normalize_text(phrase) not in normalized_transcript]
    assert not missing, {
        "missing_expected_phrases": missing,
        "expected_phrases": EXPECTED_PHRASES,
        "transcript_preview": transcript[:1000],
    }
