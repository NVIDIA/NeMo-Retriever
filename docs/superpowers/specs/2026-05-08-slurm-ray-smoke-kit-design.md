# SLURM Ray Smoke Kit Design

## Purpose

Provide a small, repo-contained handoff kit for internal engineers who want to prove that NeMo-Retriever can run against an externally started, intra-cluster Ray cluster inside their own SLURM environment.

The kit should preserve the successful experiment flow from ComputeLab while making the cluster-specific parts explicit and editable. A user should not need to hand-edit a large sbatch file for a normal run.

## Assumptions

- The user is an internal engineer with SSH access to one or more SLURM login aliases.
- The chosen cluster has a writable shared scratch path visible from the login node and allocated compute nodes.
- Compute nodes can install packages from the internet during the experiment.
- The user can run from a NeMo-Retriever source checkout, including the temporary tokenizer patch in this worktree until it lands upstream.
- The first supported workload is a CPU-only text smoke. GPU/model-heavy paths are explicitly out of scope for the initial kit.

## Recommended Flow

1. Check out or apply the SLURM-specific NeMo-Retriever worktree/branch that contains the tokenizer patch.
2. Copy a cluster profile template, such as `cluster.example.env`, to `cluster.<name>.env`.
3. Fill in the SSH alias, shared scratch root, SLURM partition/constraint/GRES settings, node count, and time limit.
4. Run `submit.sh cluster.<name>.env`.
5. The launcher validates the profile, stages the source checkout, writes a generated sbatch file into the run directory, submits it, and prints the job ID and artifact directory.
6. The user watches `squeue` and `sacct`.
7. On completion, the user inspects `summary.env`, `nemo_executor_smoke.result.json`, and `nemo_graph_ingestor_smoke.result.json`.

## Components

### `README.md`

The user-facing entry point. It should explain prerequisites, the temporary patch requirement, quickstart commands, expected successful output, and where to find artifacts.

### `cluster.example.env`

A compact, editable cluster profile. It should contain only values a new cluster user is expected to change:

- `CLUSTER_ALIAS`
- `SCRATCH_ROOT`
- `SBATCH_SELECTOR`
- `NODES`
- `NTASKS_PER_NODE`
- `CPUS_PER_TASK`
- `TIME_LIMIT`
- optional fixed Ray port block

The profile should allow multiline `SBATCH_SELECTOR` content so the launcher can support clusters that need partitions, constraints, accounts, QOS, or GPU GRES lines.

### `submit.sh`

The local launcher. It should:

- load and validate a cluster profile
- require `nemo_retriever/pyproject.toml` under the selected source root
- create a timestamped run directory on shared scratch
- tar and stage the local `nemo_retriever` package source
- copy smoke scripts and generated support files
- generate `job.sbatch` from the profile
- submit the job over SSH
- print `cluster`, `run_id`, `run_dir`, and `job_id`

The launcher should generate the sbatch file rather than asking the user to edit sbatch directly. If users need to edit the generated sbatch for normal use, the kit is missing a profile knob.

### `templates/job.sbatch.sh`

The generated job body. It should:

- allocate two nodes by default
- bootstrap identical compute-local uv, CPython 3.12, and virtualenvs under `/tmp/$USER/nemo-ray-$SLURM_JOB_ID`
- export stable `RETRIEVER_BUILD_DATE` and `RETRIEVER_BUILD_NUMBER` so dynamic package metadata cannot change during wheel construction
- create a tiny local tokenizer in the run directory
- install NeMo-Retriever from staged source plus `ray[data,serve]==2.55.1` and tokenizer-only dependencies
- start Ray head and worker as blocking `srun --overlap` steps so SLURM does not clean up daemonized Ray processes when a short step exits
- run the direct `RayDataExecutor` smoke
- run the `GraphIngestor` smoke
- stop Ray on all allocated nodes in a trap
- write summary and machine-readable artifacts

### `smokes/nemo_executor_smoke.py`

The direct proof that NeMo graph execution can use externally started Ray. It should:

- build a `Graph` with explicit node names
- pin text split/probe stages to the `nemo_worker` resource
- pin text chunk/probe stages to the `nemo_head` resource
- assert the stage host lists differ
- write a result JSON containing Ray resources, live nodes, row count, columns, host annotations, and a small sample

### `smokes/nemo_graph_ingestor_smoke.py`

The user-facing API proof. It should:

- run `GraphIngestor(run_mode="batch", ray_address=...)`
- process the same tiny text corpus
- use `node_overrides` to prove custom Ray resources flow through the high-level path
- write result JSON containing Ray resources, live nodes, row count, columns, and a small sample

### `PATCH.md`

A short explanation of the temporary source patch needed for this handoff:

- local tokenizer paths must skip pinned HuggingFace revision lookup
- Ray/Pandas binary payload variants should be normalized before text splitting
- this patch is required only until the fixes land upstream or the smoke kit moves to a released version containing them

### `triage.md`

Known failure modes and first checks:

- dynamic NeMo version changes during uv wheel build
- `ray start` succeeds but worker cannot connect because SLURM cleaned up short-lived daemon steps
- bare `env` resolves to an unexpected user path on compute nodes
- Ray Data stalls due to CPU over-reservation on a tiny proof cluster
- local tokenizer path fails pinned revision lookup
- scratch path exists on login but not compute nodes
- cluster queue remains pending on priority

## Data Flow

```text
local source checkout
  -> submit.sh
  -> tarball in run_dir/source
  -> generated job.sbatch
  -> node-local uv/Python/venv on each allocated node
  -> Ray head on node 0 + Ray worker on node 1
  -> direct RayDataExecutor smoke
  -> GraphIngestor smoke
  -> JSON/parquet/log artifacts in shared scratch
```

## Error Handling

The launcher should fail early for missing config, missing source root, unsupported profile fields, or failed staging commands.

The sbatch job should use `set -euo pipefail`, write progress to `logs/job.log`, and keep per-stage logs for setup, Ray start, Ray status, each smoke, and Ray stop. The cleanup trap should run even after smoke failures.

Smoke scripts should raise on empty outputs, missing host annotations, same-host direct stage placement, missing live Ray nodes, or missing custom Ray resources.

## Testing

Local validation:

- shell-parse `submit.sh`
- shell-parse generated `job.sbatch`
- py-compile changed text modules
- run focused unit tests for local tokenizer path handling and binary payload coercion when dependencies are available

Cluster validation:

- successful `sacct` state for the sbatch job
- `summary.env` ends with `status=succeeded`
- direct result JSON has `status=succeeded`, `rows > 0`, two alive Ray nodes, `nemo_head`, `nemo_worker`, and different split/chunk host lists
- GraphIngestor result JSON has `status=succeeded`, `rows > 0`, two alive Ray nodes, `nemo_head`, and `nemo_worker`

## Scope Boundaries

In scope:

- one-cluster, two-node SLURM Ray proof
- CPU-only text pipeline
- source-checkout staging
- generated sbatch
- shared-scratch artifacts
- cluster profile template
- triage guide based on known experiment failures

Out of scope:

- cross-cluster Ray
- local GPU model actors
- vLLM, torch, Nemotron model packages
- remote NIM endpoint credentials
- PDF/image/audio/video pipelines
- object spilling or long-running throughput tests
- production scheduler abstraction

## Open Follow-Up

DLCluster should be validated with the fixed launcher once the queued job starts. Until that passes, the kit can claim ComputeLab success and DLCluster readiness-to-run, not two-cluster completion.
