# Production checklist

Use this checklist before you run NeMo Retriever Library in production. Pair it with [Troubleshoot](troubleshoot.md) and your internal security review.

**Security and access**

- [ ] API keys and secrets follow least privilege ([Get your API key](api-keys.md), [Environment variables](environment-config.md)).
- [ ] Network policies match hosted versus self-hosted NIM choices ([When to use NVIDIA-hosted NIMs](hosted-nims-when-to-use.md), [When to self-host NIMs](self-host-nims-when-to-use.md)).

**Operations**

- [ ] Scaling model matches workload ([Scaling modes](scaling-modes.md)).
- [ ] Logging and Ray workers are observable ([Ray logging](ray-logging.md)).
- [ ] Telemetry is wired for your stack ([Telemetry](telemetry.md)).

**Quality and capacity**

- [ ] Benchmarks or load tests cover expected concurrency ([Benchmarking](benchmarking.md)).
- [ ] Dataset-specific throughput expectations are set ([Throughput is dataset-dependent](throughput-is-dataset-dependent.md)).
