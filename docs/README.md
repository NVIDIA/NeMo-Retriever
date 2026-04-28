# NeMo Retriever documentation (MkDocs)

This directory builds the **NeMo Retriever Library** static site with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Configuration is **`mkdocs.yml`** only (flat numbered nav).

## Install dependencies (once)

From the **repository root**:

```bash
pip install -r docs/requirements.txt
pip install -e ./nemo_retriever
```

## Build locally

From **`docs/`** (strict mode matches CI):

```bash
cd docs
mkdocs build -f mkdocs.yml --strict
```

The site is written to **`docs/site/`** (static HTML). You can open `site/index.html` or serve that directory with any static file server.

## Makefile (optional)

From the repository root:

```bash
make -C docs nrl-github-pages
```

This runs `mkdocs build -f mkdocs.yml --strict` with `cwd` set to `docs/`.

## GitHub Pages

On **`NVIDIA/NeMo-Retriever`**, [`.github/workflows/nrl-docs-github-pages.yml`](../.github/workflows/nrl-docs-github-pages.yml) installs the same dependencies, sets `SITE_URL` via `actions/configure-pages`, runs **`mkdocs build -f mkdocs.yml --strict`**, then publishes the **`docs/site`** artifact to GitHub Pages (`deploy-pages`).

Forks and other repositories get a downloadable **`nrl-docs-site`** artifact instead of a Pages deploy.
