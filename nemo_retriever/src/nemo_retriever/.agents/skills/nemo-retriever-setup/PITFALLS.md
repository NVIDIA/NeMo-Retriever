# Setup Pitfalls

## Python Version

Use Python 3.12. Older or newer Python versions can fail dependency resolution
or import checks because the package metadata requires `>=3.12,<3.13`.

## Missing Installed Surface

`retriever`, `nemo-retriever`, and `nemo_retriever` are not interchangeable
command names. The public CLI command is `retriever`; the Python import package
is `nemo_retriever`; the distribution name is `nemo-retriever`.

## Source Checkout Versus Installed Package

`uv run --project nemo_retriever retriever ...` is a developer-checkout fallback.
For installed-package validation, install the package into an isolated
environment and run `retriever --help` without relying on the source tree.

## Optional Extras

The base install is enough for remote NIM workflows. Local GPU inference needs
the `local` extra. Audio/video and SVG workflows need `multimedia` plus system
dependencies. QA generation/judging needs `llm`.

## System Dependencies

`ffmpeg-python` and `nemo-retriever[multimedia]` do not install the `ffmpeg` and
`ffprobe` binaries. Install those through the operating system or use a service
image/cluster configuration that provides them.

## Model Downloads

Local inference may download large HuggingFace assets on first use. Route caches
to `~/models` for reproducible agent work and avoid writing model assets into
the repository.

## Remote Credentials

Hosted NIM endpoints need `NVIDIA_API_KEY`. Missing keys should be reported as a
setup gap, not as an ingest/query failure.
