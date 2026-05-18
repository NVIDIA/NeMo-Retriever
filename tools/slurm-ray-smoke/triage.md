# SLURM Ray Smoke Triage

## Dynamic build version changes during install

Symptom: uv builds NeMo-Retriever more than once and the dynamic version changes during the job.

First check:

```bash
grep -R "RETRIEVER_BUILD" logs/setup.log
```

Expected fix: `job.sbatch` exports stable `RETRIEVER_BUILD_DATE` and `RETRIEVER_BUILD_NUMBER` before package installation.

## Ray start succeeds but workers cannot connect

Symptom: `ray start` reports success, but later `ray status` cannot see the worker.

First check:

```bash
grep -R "ray start\|Failed\|Traceback" logs/ray-*.log
```

Expected fix: Ray head and workers run under blocking `ray start --block` commands launched by background `srun --overlap` steps, and worker startup waits for head readiness.

## Wrong env binary on compute nodes

Symptom: uv installer or shebangs resolve a user-path `env` that is not valid on compute nodes.

First check:

```bash
grep -R "/usr/bin/env" job.sbatch logs/setup.log
```

Expected fix: installer commands use `/usr/bin/env`, while generated job scripts use absolute uv and venv paths for Python and Ray commands.

## Ray Data stalls due to CPU overreservation

Symptom: the Ray cluster is alive, but smoke output stalls after setup.

First check:

```bash
cat logs/ray-status-final.log
tail -n 100 logs/smokes.log
```

Expected fix: smoke overrides use low `num_cpus`, `concurrency=1`, and small custom resource fractions.

## Local tokenizer path fails revision lookup

Symptom: smoke logs show Hugging Face revision lookup for the run-local tokenizer path.

First check:

```bash
grep -R "get_hf_revision\|revision" logs/smokes.log
```

Expected fix: apply the local tokenizer path patch described in `PATCH.md`.

## Scratch path missing on compute nodes

Symptom: setup cannot read `source/nemo_retriever-src.tar.gz` or write run artifacts.

First check:

```bash
grep -R "tar\|No such file" logs/setup.log logs/slurm-*.err
```

Expected fix: set `SCRATCH_ROOT` to a filesystem mounted on both the login node and allocated compute nodes.

## Queue pending on priority

Symptom: `squeue` shows the job pending and `sacct` has no terminal state.

First check:

```bash
JOB_ID=123456
squeue -j "$JOB_ID" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

Expected fix: wait for the scheduler or change `SBATCH_SELECTOR` to a partition, account, or QOS with available capacity.
