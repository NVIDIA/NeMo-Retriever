# Temporary Source Patch

This handoff currently requires two temporary text-ingestion fixes in the NeMo-Retriever source checkout submitted by `submit.sh`.

## Local Tokenizer Paths

The smoke creates a tiny tokenizer in the run directory and passes that local path as `tokenizer_model_id`. `_get_tokenizer()` must skip `get_hf_revision()` when `model_id` already exists as a local path, then call `AutoTokenizer.from_pretrained()` with the local path, cache directory, and `trust_remote_code=True`.

## Ray/Pandas Byte Payloads

Ray Data and Pandas can pass text payloads to split actors as several binary shapes. `TxtSplitCPUActor.process()` must normalize `bytes`, `bytearray`, `memoryview`, Arrow-style values with `.as_py()`, and objects with `.tobytes()` before calling `txt_bytes_to_chunks_df()`.

## Removal Criteria

Keep using the SLURM smoke branch until both fixes land upstream or appear in a released NeMo-Retriever version. After that, update this note and the README prerequisite before pointing the kit at the released package.
