# OCR package builds and releases

NeMo Retriever Library (NRL) uses the
[Hugging Face build workflow](../.github/workflows/huggingface-nightly.yml)
to build optical character recognition (OCR) source from
[`nvidia/nemotron-ocr-v2`](https://huggingface.co/nvidia/nemotron-ocr-v2)
and publishes the distribution as `nemotron-ocr` on public PyPI.
Use this workflow to qualify a new Torch build before publishing it.

## Select the build stack

The workflow provides the following OCR build profiles.

| `ocr_build_stack` | Torch | torchvision | Nightly version family |
| --- | --- | --- | --- |
| `torch211` | `2.11.0` | `0.26.0` | `2.0.2.dev…` |
| `torch213` | `2.13.0` | `0.28.0` | `2.1.0.dev…` |

Scheduled runs and manual runs that retain the default profile use `torch211`.
The `torch213` profile requires `package=nemotron-ocr-v2` and an exact,
40-character Hugging Face commit in `ocr_source_revision`.
The builder checks out that commit before modifying package metadata.

Both profiles use CUDA 13.0 and build Linux x86_64 and aarch64 wheels for
Python 3.11, 3.12, and 3.13, plus one source distribution.
The native extension is compiled against the selected Torch stack.
Its package requirements constrain Torch and torchvision to the corresponding
minor versions, such as `torch~=2.13.0` and `torchvision~=0.28.0`.

## Build and qualify artifacts

In GitHub Actions, select **huggingface-nightly-build-and-publish**, choose the
candidate branch under **Run workflow**, and set these inputs.

| Input | Value |
| --- | --- |
| `package` | `nemotron-ocr-v2` |
| `release_type` | `nightly` |
| `release_version` | Leave empty. |
| `upload_to` | `none` |
| `ocr_build_stack` | `torch213` |
| `ocr_source_revision` | The full HF commit to qualify. |

The workflow defaults `upload_to` to `pypi`; explicitly select `none` for
qualification builds.
Download each `dist-nemotron-ocr-v2-<arch>-<python-tag>` artifact after the run
completes, such as `dist-nemotron-ocr-v2-x86_64-cp312`.
Inspect its `build-info.json` and retain it with the test results.
Confirm that every matrix leg used the intended source commit and build stack.

Workflow checks validate package contents, metadata, and Python extension ABI
tags before upload.
The workflow explicitly uses core metadata 2.4 for wheels and source distributions
to keep the release format stable across build-backend updates.
Run GPU inference checks on the produced wheels with the intended runtime.
Cover detector, recognizer, and relational output, batching, OCR accuracy, and
performance on the supported hardware you need to qualify.
A passing local Python 3.12 build on one GPU does not qualify every workflow
wheel or GPU architecture.

## Publish the qualified build

Run the same workflow with the qualified source revision and build profile,
setting `upload_to=pypi`.
The nightly version includes a timestamp, so this run builds a new version.
Compare its `build-info.json` with the qualification artifacts before adopting
the published package.
Publication uses the repository's existing `PYPI_API_TOKEN` secret.

For a stable release, select `release_type=stable` and supply a fresh
`release_version` while retaining the qualified stack and source revision.
Stable releases require selecting one package.

The Torch 2.13 build is a new version of `nemotron-ocr`, with its own runtime
requirements.
Older releases remain available for Torch 2.11 installations.
Use an exact version pin when testing or adopting the published prerelease.
Repeat the GPU smoke test with that version before updating consumers.
Publishing this package does not update NRL runtime dependency pins or its
lockfile; coordinate those changes with validation of the complete inference
stack.
