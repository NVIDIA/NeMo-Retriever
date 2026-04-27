# kheiss/NRLbuild — completed work summary

Branch tip relative to `origin/main` (merge-base `26091df7`): documentation, GitHub Pages, and CI consolidation for **NeMo Retriever Library (NRL)** publishing, plus merged library and evaluation work from the same integration line.

Use `git log 26091df7..HEAD --oneline` for the full commit list.

---

## NeMo Retriever Library documentation and GitHub Pages

### Single MkDocs configuration

- **One canonical config:** `docs/mkdocs.yml` drives both NVIDIA docs and GitHub Pages. The duplicate **`docs/mkdocs.nrl-github-pages.yml`** was removed.
- **Build entrypoints:** `docs/Makefile`, `.github/workflows/build-docs.yml`, and `.github/workflows/docs-deploy.yml` were aligned to use `mkdocs.yml` (including **`mkdocs build --strict`** where applicable).

### GitHub Pages workflow (`nrl-docs-github-pages.yml`)

- **NRL-only** MkDocs + Material build and deploy to GitHub Pages (no full Docker + Sphinx nv-ingest API dump on this path).
- **Action pinning:** third-party actions pinned to full commit SHAs from official tagged releases (security / Greptile review).
- **Pre-deploy checks:** prints the MkDocs nav (`docs/scripts/print_nrl_mkdocs_nav.py`); runs **`scan_non_nrl_doc_references.py`** with an excerpt in the job step summary and the full report as a workflow artifact.
- **Concurrency and ownership:** adjustments so this workflow remains the intended Pages publisher and is not overwritten or duplicated by legacy Docker-based doc jobs (see history in #1922, #1925).

### Table of contents and redirects

- **§3 Deployment:** one nav page, **[`deployment-options.md`](../docs/docs/extraction/deployment-options.md)** — compares local library use, Helm/Kubernetes, notebooks/API keys, and performance-oriented links. Legacy topics remain in the repo for parity but are **excluded from the build** and covered by **`redirect_maps`** toward `deployment-options.md`: `choose-your-path.md`, `hosted-nims-when-to-use.md`, `self-host-nims-when-to-use.md`, `quickstart-guide.md`, and `helm.md`.
- **§4 Core workflows:** **Audio & Video Ingestion** nests **`audio-video.md`** and **`workflow-video-ocr.md`** (OCR of video frames).
- **NVIDIA Blueprints / E2E RAG:** removed the redundant nav stub **`workflow-e2e-blueprints.md`**; **redirect** to **`resources-links.md`**; file listed in **`exclude_docs`**. **Cross-links:** [`overview.md`](../docs/docs/extraction/overview.md) Related Topics → `resources-links.md`; **`resources-links.md`** See also → `overview.md`.

### Content and strict-build fixes

- Removed repeated boilerplate “NeMo Retriever Library” notes across many extraction pages (#1904 and follow-ups).
- **`nemo_retriever/docs/cli/`** README and links updated for deployment and workflow paths.
- **`nemo_retriever/.../retriever.py`:** doctest adjustment for **mkdocstrings** / **`--strict`** (#1925).

### Reviews

- **Greptile** feedback addressed in dedicated commits (`37e0ed9b`, `8cab4d64`, and related).
- **Randy’s review:** follow-up captured in **`e40a1626`** (*Updating NRL GitHub pages following Randy's review and feedback*).

---

## Library and product changes on the same branch line

High-level themes from merged commits on this branch (see full diff with `git diff --stat origin/main..HEAD`):

- **Pipeline / CLI:** pipeline subcommand, evaluation CLI updates, and bundled CLI documentation under `nemo_retriever/docs/cli/`.
- **Embeddings:** vLLM-backed text/VLM embedder (#1494); Llama Nemotron embedder paths and HF support.
- **Tabular / SQL:** embedder parameters in generated SQL (#1918); query comparison tooling and tests (#1928).
- **Agentic retrieval / bench:** AbstractOperators pattern (#1784); additional graph operators and expanded tests.
- **Docs:** default Llama tokenizer for token-based chunking (#1914); `chunking.md` and Python API cross-links.
- **Examples / tests:** `graph_pipeline.py` refactor; benchmark sample queries; assorted lockfile and `.gitignore` updates.

---

## Verification commands

```bash
cd docs
python -m mkdocs build -f mkdocs.yml --strict
```

NRL Pages workflow triggers on pushes to `main` when `docs/**`, `nemo_retriever/**`, or `nrl-docs-github-pages.yml` change (plus schedule / `workflow_dispatch` per workflow file).
