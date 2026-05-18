# NeMo-Retriever SLURM Ray Smoke Kit

This kit submits a two-node SLURM job that starts Ray inside one allocation and verifies NeMo-Retriever can use that external Ray cluster through both the direct `RayDataExecutor` path and `GraphIngestor`.

The local launcher stages the current NeMo-Retriever source tree and smoke scripts to shared scratch, generates `job.sbatch`, submits it over SSH, then prints the run metadata needed to watch the job and inspect artifacts.

## Prerequisites

- SSH access to a SLURM login alias.
- A shared scratch path visible from the login node and allocated compute nodes.
- Compute nodes can download uv, CPython, Ray, and tokenizer dependencies during setup.
- A NeMo-Retriever source checkout containing the temporary text tokenizer and byte payload patches described in `PATCH.md`.

## Basic Usage

### 1. Use the SLURM smoke branch

Start from a NeMo-Retriever checkout that includes the temporary text-ingestion fixes described in `PATCH.md`.

```bash
cd /path/to/NeMo-Retriever
git status --short --branch
```

If you are sharing this work before the fixes land upstream, have the user check out this branch or apply the equivalent patch first.

### 2. Create a cluster profile

Copy the example profile and name it after the SLURM login alias or site.

```bash
cp tools/slurm-ray-smoke/cluster.example.env tools/slurm-ray-smoke/cluster.my-slurm.env
${EDITOR:-vi} tools/slurm-ray-smoke/cluster.my-slurm.env
```

Set these values first:

```bash
CLUSTER_ALIAS=slurm-login
SCRATCH_ROOT=/shared/scratch/${USER}/ray-xcluster
NODES=2
NTASKS_PER_NODE=1
CPUS_PER_TASK=8
TIME_LIMIT=00:30:00
```

Then replace `SBATCH_SELECTOR` with the site-specific scheduler lines, for example:

```bash
SBATCH_SELECTOR=$(cat <<'SBATCH_SELECTOR_EOF'
#SBATCH --partition=debug
#SBATCH --account=research
SBATCH_SELECTOR_EOF
)
```

Leave the Ray ports at their defaults unless the cluster has a known conflict. If the NeMo-Retriever source tree is somewhere other than `./nemo_retriever` under the repo root, set `SOURCE_ROOT=/absolute/path/to/nemo_retriever`.

Do not commit cluster profiles with account, partition, or scratch details unless they are intended examples.

### 3. Submit the smoke job

Run the launcher from the repository root:

```bash
tools/slurm-ray-smoke/submit.sh tools/slurm-ray-smoke/cluster.my-slurm.env
```

The launcher validates the profile, stages the selected source tree and smoke scripts to shared scratch, renders `job.sbatch` into the run directory, submits it over SSH, and prints run metadata:

```text
cluster=slurm-login
run_id=20260508T153353Z-slurm-login-nemo-intracluster-ray
run_dir=/shared/scratch/user/ray-xcluster/nemo-intracluster/runs/20260508T153353Z-slurm-login-nemo-intracluster-ray
job_id=123456
```

Keep `run_dir` and `job_id`; the rest of the workflow uses them.

### 4. Watch SLURM

```bash
CLUSTER_ALIAS=slurm-login
JOB_ID=123456

ssh "$CLUSTER_ALIAS" "squeue -j '$JOB_ID' -o '%.18i %.9P %.8j %.8u %.2t %.19S %.10M %.6D %R'"
ssh "$CLUSTER_ALIAS" "sacct -j '$JOB_ID' --format=JobID,State,ExitCode,Elapsed,NodeList"
```

The expected terminal state is `COMPLETED` with exit code `0:0`. If the job remains pending, use the reason from `squeue` and the checks in `triage.md`.

### 5. Inspect the artifacts

```bash
CLUSTER_ALIAS=slurm-login
RUN_DIR=/shared/scratch/user/ray-xcluster/nemo-intracluster/runs/20260508T153353Z-slurm-login-nemo-intracluster-ray

ssh "$CLUSTER_ALIAS" "cat '$RUN_DIR/summary.env'"
ssh "$CLUSTER_ALIAS" "RUN_DIR='$RUN_DIR' python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR'])
for name in ('nemo_executor_smoke.result.json', 'nemo_graph_ingestor_smoke.result.json'):
    payload = json.loads((run_dir / name).read_text())
    print(name, payload['status'], 'rows=', payload['rows'])
    print('resources=', payload['ray']['cluster_resources'])

direct = json.loads((run_dir / 'nemo_executor_smoke.result.json').read_text())
print('split_hosts=', direct['split_hosts'])
print('chunk_hosts=', direct['chunk_hosts'])
PY"
```

Success means `summary.env` has `status=succeeded`, both result JSON files have `status=succeeded` and `rows=3`, Ray reports at least two live nodes, and the direct smoke shows different `split_hosts` and `chunk_hosts`.

## Profile Fields

- `CLUSTER_ALIAS`: SSH alias for the SLURM login node.
- `SCRATCH_ROOT`: shared scratch root where run directories are created.
- `SBATCH_SELECTOR`: multiline `#SBATCH` lines for partition, account, QOS, constraints, or similar site selectors.
- `NODES`: node count. Use `2` for this smoke.
- `NTASKS_PER_NODE`: task count per node. Use `1`.
- `CPUS_PER_TASK`: CPU cores reserved per Ray process.
- `TIME_LIMIT`: SLURM wall time such as `00:30:00`.
- `RAY_PORT`, `RAY_DASHBOARD_PORT`, `RAY_CLIENT_SERVER_PORT`: Ray ports used inside the allocation.
- `SOURCE_ROOT`: optional path to the `nemo_retriever` package directory to stage. Defaults to `./nemo_retriever` under the repo root.

The launcher generates `job.sbatch`. If a normal run requires editing generated sbatch by hand, add a profile field first so the run stays reproducible.

## Successful Artifacts

On success, inspect these files under `run_dir`:

- `summary.env`: contains `status=succeeded`, `job_id`, `run_dir`, `head_node`, `head_ip`, `ray_address`, and allocated nodes.
- `nemo_executor_smoke.result.json`: direct `RayDataExecutor` proof with `status=succeeded`, `rows=3`, live Ray nodes, `nemo_head` and `nemo_worker` resources, and different `split_hosts` and `chunk_hosts`. The smoke records those host names inside smoke-local actor subclasses.
- `nemo_graph_ingestor_smoke.result.json`: `GraphIngestor(run_mode="batch", ray_address=...)` proof with `status=succeeded`, `rows=3`, live Ray nodes, and `nemo_head` and `nemo_worker` resources.
- `logs/`: setup, Ray start/status/stop, smoke, and SLURM stdout/stderr logs.

## Known Baseline

ComputeLab kit validation job `2091090` completed on `ipp1-1135` and `ipp1-1136`. The direct smoke produced three rows with split work on `ipp1-1136`, chunk work on `ipp1-1135`, and both custom resources present.
