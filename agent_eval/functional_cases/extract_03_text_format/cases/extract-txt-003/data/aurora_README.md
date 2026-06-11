# Aurora — Inference Platform

> **Status:** GA · **Owner:** Platform Engineering · **Last reviewed:** 2026-11-04

Aurora is the internal inference platform powering all NeMo Retriever hosted endpoints.

## Quick start

```bash
# Set your API key
export AURORA_API_KEY="aurora-..."

# Hello-world request
curl -H "Authorization: Bearer $AURORA_API_KEY" https://aurora.example.com/v1/embed      -d '{"inputs":["hello"]}'
```

## Capacity

| Region | GPUs | SLO p99 |
|---|---:|---:|
| us-east-2 | 512 H100 | 280ms |
| us-west-2 | 384 H100 | 290ms |
| eu-central-1 | 192 H200 | 310ms |
| ap-northeast-1 | 128 H200 | 340ms |

## Escalation

For incidents page **#aurora-oncall** in Slack. SLA target is 15-min response, 1-hour mitigation.

## Related docs

See [architecture.html](../html/architecture.html) and [runbook.html](../html/runbook.html).
