# NeMo-Retriever SLURM Ray Smoke Kit

This kit submits a two-node SLURM job that starts Ray inside one allocation and verifies NeMo-Retriever can use that external Ray cluster through both the direct `RayDataExecutor` path and `GraphIngestor`.

The local launcher stages the current NeMo-Retriever source tree and smoke scripts to shared scratch, generates `job.sbatch`, submits it over SSH, then prints the run metadata needed to watch the job and inspect artifacts.

## Prerequisites

- SSH access to a SLURM login alias.
- A shared scratch path visible from the login node and allocated compute nodes.
- Compute nodes can download uv, CPython, Ray, and tokenizer dependencies during setup.
- A NeMo-Retriever source checkout containing the temporary text tokenizer and byte payload patches described in `PATCH.md`.

## Quickstart

```bash
cd /path/to/NeMo-Retriever
cp tools/slurm-ray-smoke/cluster.example.env tools/slurm-ray-smoke/cluster.my-slurm.env
${EDITOR:-vi} tools/slurm-ray-smoke/cluster.my-slurm.env
tools/slurm-ray-smoke/submit.sh tools/slurm-ray-smoke/cluster.my-slurm.env
```

Example submit output:

```text
cluster=slurm-login
run_id=20260508T153353Z-slurm-login-nemo-intracluster-ray
run_dir=/shared/scratch/user/ray-xcluster/nemo-intracluster/runs/20260508T153353Z-slurm-login-nemo-intracluster-ray
job_id=123456
```

## Profile Fields

- `CLUSTER_ALIAS`: SSH alias for the SLURM login node.
- `SCRATCH_ROOT`: shared scratch root where run directories are created.
- `SBATCH_SELECTOR`: multiline `#SBATCH` lines for partition, account, QOS, constraints, or similar site selectors.
- `NODES`: node count. Use `2` for this smoke.
- `NTASKS_PER_NODE`: task count per node. Use `1`.
- `CPUS_PER_TASK`: CPU cores reserved per Ray process.
- `TIME_LIMIT`: SLURM wall time such as `00:30:00`.
- `RAY_PORT`, `RAY_DASHBOARD_PORT`, `RAY_CLIENT_SERVER_PORT`: Ray ports used inside the allocation.

The launcher generates `job.sbatch`. If a normal run requires editing generated sbatch by hand, add a profile field first so the run stays reproducible.

## Watching the Job

```bash
ssh slurm-login 'squeue -j 123456'
ssh slurm-login 'sacct -j 123456 --format=JobID,State,ExitCode,Elapsed,NodeList'
```

## Successful Artifacts

On success, inspect these files under `run_dir`:

- `summary.env`: contains `status=succeeded`, `job_id`, `run_dir`, `head_node`, `head_ip`, `ray_address`, and allocated nodes.
- `nemo_executor_smoke.result.json`: direct `RayDataExecutor` proof with `status=succeeded`, `rows=3`, live Ray nodes, `nemo_head` and `nemo_worker` resources, and different `split_hosts` and `chunk_hosts`. The smoke records those host names inside smoke-local actor subclasses.
- `nemo_graph_ingestor_smoke.result.json`: `GraphIngestor(run_mode="batch", ray_address=...)` proof with `status=succeeded`, `rows=3`, live Ray nodes, and `nemo_head` and `nemo_worker` resources.
- `logs/`: setup, Ray start/status/stop, smoke, and SLURM stdout/stderr logs.

## Known Baseline

ComputeLab kit validation job `2091090` completed on `ipp1-1135` and `ipp1-1136`. The direct smoke produced three rows with split work on `ipp1-1136`, chunk work on `ipp1-1135`, and both custom resources present.
