# NeMo Retriever Development Compose Helpers

These Compose files are local development helpers for feature-specific services. They do not start the NeMo Retriever service or any legacy ingestion runtime.

Run commands from the repository root.

## Local Judge

The judge helper starts an OpenAI-compatible Nemotron NIM for `retriever skill-eval` runs that use a local judge endpoint.

Set `NGC_API_KEY` or `NIM_NGC_API_KEY` before starting this helper.

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge
```

Then point `judge.api_base` at `http://localhost:8000/v1` in your skill-eval config.

## Neo4j

The Neo4j helper starts a local database for tabular/graph development.

```bash
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j
```

Set `NEO4J_PASSWORD` in your environment or `.env` file before starting this helper. `NEO4J_USERNAME` defaults to `neo4j`.
