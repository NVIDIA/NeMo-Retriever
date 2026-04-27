# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Define paths
SPHINX_BUILD_DIR=sphinx_docs/build
SPHINX_SOURCE_DIR=sphinx_docs/source
SPHINX_OUTPUT_DIR=docs/extraction/api-docs

# Check for the nv_ingest module
INGEST_PYTHON_MODULE_EXISTS := $(shell python -c "import importlib.util; exit(0 if importlib.util.find_spec('nv_ingest') else 1)" && echo 1 || echo 0)
PYTHONPATH=$(CURDIR)/../src:$(CURDIR)/../api/src:$(CURDIR)/../client/src:$(CURDIR)/../nemo_retriever/src


# Default target
.PHONY: all
all: docs

# Run apidoc against nv-ingest, nv-ingest-api, and nv-ingest-client codebase
.PHONY: sphinx-apidoc
sphinx-apidoc:
	@echo "🏃 Running sphinx-apidoc against nv-ingest"
	sphinx-apidoc -o sphinx_docs/source/nv-ingest ../src/nv_ingest

	@echo "🏃 Running sphinx-apidoc against nv-ingest-api"
	sphinx-apidoc -o sphinx_docs/source/nv-ingest-api ../api/src/nv_ingest_api

	@echo "🏃 Running sphinx-apidoc against nv-ingest-client"
	sphinx-apidoc -o sphinx_docs/source/nv-ingest-client ../client/src/nv_ingest_client

	@echo "🏃 Running sphinx-apidoc against nemo-retriever"
	sphinx-apidoc -o sphinx_docs/source/nemo-retriever ../nemo_retriever/src/nemo_retriever

# Generate nv-ingest openapi specification schema
.PHONY: generate-openapi-schema
generate-openapi-schema:
	@echo "📖 Generating nv-ingest openapi specification ..."
	@if [ "$(INGEST_PYTHON_MODULE_EXISTS)" -eq "1" ]; then \
		{ \
			echo "nv-ingest module found."; \
			python ./scripts/generate_openapi_docs.py --output ./sphinx_docs/source/openapi.yaml; \
		} \
	else \
		echo "Module not found. Adjust PYTHONPATH to: $(PYTHONPATH)"; \
		PYTHONPATH=$(PYTHONPATH) python ./scripts/generate_openapi_docs.py --output ./sphinx_docs/source/openapi.yaml; \
	fi

# Build Sphinx documentation
.PHONY: sphinx
sphinx:
	@echo "📖 Building Sphinx documentation..."
	sphinx-build -b html $(SPHINX_SOURCE_DIR) $(SPHINX_BUILD_DIR)/html

# Copy Sphinx HTML output to the final location
.PHONY: copy-sphinx
copy-sphinx: sphinx
	@echo "📂 Copying Sphinx docs to $(SPHINX_OUTPUT_DIR)..."
	mkdir -p $(SPHINX_OUTPUT_DIR)
	cp -r $(SPHINX_BUILD_DIR)/html/* $(SPHINX_OUTPUT_DIR)/

# Run MkDocs: same config and flags as GitHub Pages (mkdocs.yml).
.PHONY: mkdocs
mkdocs:
	@echo "📚 Building MkDocs site (mkdocs.yml, strict)..."
	mkdocs build -f mkdocs.yml --strict

# Alias for CI/docs comments; identical to mkdocs.
.PHONY: docs-ci
docs-ci: mkdocs

# NeMo Retriever Library docs (GitHub Pages): alias for mkdocs (flat 13-section nav in mkdocs.yml).
.PHONY: nrl-github-pages
nrl-github-pages: mkdocs

# Full docs pipeline
.PHONY: docs
docs: sphinx-apidoc generate-openapi-schema sphinx copy-sphinx mkdocs
	@echo "✅ Documentation build complete!"

# Clean up built docs
.PHONY: clean
clean:
	@echo "🧹 Cleaning up built documentation..."
	rm -rf $(SPHINX_BUILD_DIR) $(SPHINX_OUTPUT_DIR) site
