# SLURM Ray Smoke Kit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repo-contained SLURM handoff kit that lets an internal engineer copy one cluster profile, run one submit command, and verify NeMo-Retriever against an externally started two-node Ray cluster.

**Architecture:** Add a focused `tools/slurm-ray-smoke/` kit containing a profile template, local launcher, generated sbatch template, two Python smoke tests, and troubleshooting docs. The launcher stages the local NeMo-Retriever source tree to shared scratch, renders `job.sbatch`, submits over SSH, and prints the job ID plus artifact directory. The sbatch job bootstraps node-local Python/uv environments, starts Ray with blocking `srun --overlap` steps, runs a direct `RayDataExecutor` proof and a `GraphIngestor` proof, then writes machine-readable result JSON.

**Tech Stack:** Bash, SSH, SLURM `sbatch`/`srun`/`sacct`, uv, CPython 3.12, Ray Data 2.55.1, Pandas, Hugging Face `transformers`/`tokenizers`, NeMo-Retriever graph APIs, pytest.

---

## File Structure

- Create: `tools/slurm-ray-smoke/README.md` - user-facing quickstart, prerequisites, expected artifacts, and cluster verification flow.
- Create: `tools/slurm-ray-smoke/cluster.example.env` - compact shell profile that users copy to a file such as `cluster.my-slurm.env`.
- Create: `tools/slurm-ray-smoke/submit.sh` - local launcher that validates a profile, stages source/smokes, renders `job.sbatch`, submits over SSH, and prints run metadata.
- Create: `tools/slurm-ray-smoke/templates/job.sbatch.sh` - rendered SLURM job body with stable build metadata, node-local uv/Python setup, blocking Ray lifecycle, smoke execution, and cleanup trap.
- Create: `tools/slurm-ray-smoke/smokes/_common.py` - shared JSON, Ray-state, and DataFrame helpers used by both smoke scripts.
- Create: `tools/slurm-ray-smoke/smokes/nemo_executor_smoke.py` - direct `RayDataExecutor` proof that pins stages to `nemo_worker` and `nemo_head` custom Ray resources and asserts host separation.
- Create: `tools/slurm-ray-smoke/smokes/nemo_graph_ingestor_smoke.py` - `GraphIngestor(run_mode="batch", ray_address=...)` proof using the same local tokenizer and custom resource override.
- Create: `tools/slurm-ray-smoke/PATCH.md` - short explanation of the temporary source patch required for handoff.
- Create: `tools/slurm-ray-smoke/triage.md` - first checks for known failure modes from the ComputeLab experiment.
- Modify: `nemo_retriever/src/nemo_retriever/txt/split.py` - skip pinned Hugging Face revision lookup when `tokenizer_model_id` is a local path.
- Modify: `nemo_retriever/tests/test_txt_split.py` - unit coverage for local tokenizer path behavior.
- Modify: `nemo_retriever/src/nemo_retriever/txt/ray_data.py` - normalize Ray/Pandas binary payload variants before text splitting.
- Modify: `nemo_retriever/tests/test_actor_operators.py` - unit coverage for `memoryview`, `bytearray`, `.as_py()`, and `.tobytes()` byte payloads.

## Known-Good Baseline

Use this baseline to compare cluster validation results:

- ComputeLab fixed run: SLURM job `2090092`, state `COMPLETED`, exit `0:0`.
- Nodes: `ipp1-1211` as Ray head and `ipp1-1212` as Ray worker.
- Artifact directory: `/home/scratch.charlesb_sw/ray-xcluster/nemo-intracluster/runs/20260508T153353Z-computelab-sc-01-nemo-intracluster-ray`.
- Direct smoke result: `status=succeeded`, `rows=3`, split host `ipp1-1212`, chunk host `ipp1-1211`, two alive Ray nodes, custom resources `nemo_head=1` and `nemo_worker=1`.
- GraphIngestor result: `status=succeeded`, `rows=3`, two alive Ray nodes, custom resources `nemo_head=1` and `nemo_worker=1`.
- DLCluster fixed job `960477` was submitted but remained pending on Priority with estimated start `2026-05-09T03:38:00Z`.

### Task 1: Local Tokenizer Path Patch

**Files:**
- Modify: `nemo_retriever/tests/test_txt_split.py`
- Modify: `nemo_retriever/src/nemo_retriever/txt/split.py`

- [ ] **Step 1: Write the failing local-tokenizer test**

Add these imports near the top of `nemo_retriever/tests/test_txt_split.py` if they are not already present:

```python
import sys
import types
from pathlib import Path
```

Add this test after `test_split_text_by_tokens_max_tokens_positive`:

```python
def test_get_tokenizer_local_path_skips_revision_lookup(tmp_path: Path, monkeypatch):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    calls = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_id"] = model_id
            calls["kwargs"] = kwargs
            return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def fail_revision_lookup(model_id):
        raise AssertionError(f"unexpected revision lookup for {model_id}")

    monkeypatch.setattr("nemo_retriever.utils.hf_model_registry.get_hf_revision", fail_revision_lookup)

    _get_tokenizer(str(tokenizer_dir), cache_dir="/tmp/tokenizer-cache")

    assert calls["model_id"] == str(tokenizer_dir)
    assert calls["kwargs"]["cache_dir"] == "/tmp/tokenizer-cache"
    assert calls["kwargs"]["trust_remote_code"] is True
    assert "revision" not in calls["kwargs"]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run --project nemo_retriever pytest nemo_retriever/tests/test_txt_split.py::test_get_tokenizer_local_path_skips_revision_lookup -q
```

Expected: FAIL with `AssertionError: unexpected revision lookup for` followed by the temporary tokenizer path.

- [ ] **Step 3: Implement local path detection**

Replace `_get_tokenizer` in `nemo_retriever/src/nemo_retriever/txt/split.py` with:

```python
def _get_tokenizer(model_id: str, cache_dir: Optional[str] = None):  # noqa: ANN201
    """Lazy-load HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }
    if not Path(model_id).expanduser().exists():
        from nemo_retriever.utils.hf_model_registry import get_hf_revision

        tokenizer_kwargs["revision"] = get_hf_revision(model_id)

    return AutoTokenizer.from_pretrained(
        model_id,
        **tokenizer_kwargs,
    )
```

- [ ] **Step 4: Run tokenizer tests**

Run:

```bash
uv run --project nemo_retriever pytest nemo_retriever/tests/test_txt_split.py::test_get_tokenizer_local_path_skips_revision_lookup nemo_retriever/tests/test_txt_split.py::test_txt_file_to_chunks_df -q
```

Expected: PASS. If this environment lacks optional test dependencies, record the missing package and still run the syntax check in Step 5.

- [ ] **Step 5: Run syntax check**

Run:

```bash
python -m py_compile nemo_retriever/src/nemo_retriever/txt/split.py nemo_retriever/tests/test_txt_split.py
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Commit tokenizer patch**

Run:

```bash
git add nemo_retriever/src/nemo_retriever/txt/split.py nemo_retriever/tests/test_txt_split.py
git commit -m "fix: allow local tokenizer paths for text splitting"
```

### Task 2: Ray/Pandas Binary Payload Coercion

**Files:**
- Modify: `nemo_retriever/tests/test_actor_operators.py`
- Modify: `nemo_retriever/src/nemo_retriever/txt/ray_data.py`

- [ ] **Step 1: Write payload coercion tests**

Add these helper classes near `class TestTxtSplitActor` in `nemo_retriever/tests/test_actor_operators.py`:

```python
class _FakeArrowScalar:
    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class _FakeTobytes:
    def __init__(self, value):
        self._value = value

    def tobytes(self):
        return self._value
```

Add this parameterized test inside `class TestTxtSplitActor` after `test_process`:

```python
    @pytest.mark.parametrize(
        ("raw", "expected_bytes"),
        [
            (memoryview(b"hello"), b"hello"),
            (bytearray(b"hello"), b"hello"),
            (_FakeArrowScalar(b"hello"), b"hello"),
            (_FakeTobytes(b"hello"), b"hello"),
        ],
    )
    @patch("nemo_retriever.txt.ray_data.txt_bytes_to_chunks_df")
    def test_process_coerces_binary_payload_variants(self, mock_fn, raw, expected_bytes):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.txt"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [raw], "path": ["/a.txt"]})

        result = actor.process(df)

        mock_fn.assert_called_once_with(expected_bytes, "/a.txt", params=actor._params)
        pd.testing.assert_frame_equal(result, expected)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run --project nemo_retriever pytest nemo_retriever/tests/test_actor_operators.py::TestTxtSplitActor::test_process_coerces_binary_payload_variants -q
```

Expected: FAIL because `txt_bytes_to_chunks_df` receives a non-`bytes` object for at least one parameter case.

- [ ] **Step 3: Implement payload coercion helper**

Add this helper above `TextChunkCPUActor` in `nemo_retriever/src/nemo_retriever/txt/ray_data.py`:

```python
def _coerce_binary_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()

    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return _coerce_binary_payload(as_py())

    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes):
        return tobytes()

    return value
```

Update the `TxtSplitCPUActor.process` loop in the same file so the payload is normalized before calling `txt_bytes_to_chunks_df`:

```python
            try:
                payload = _coerce_binary_payload(raw)
                if payload is None and isinstance(text, str):
                    payload = text.encode("utf-8")
                if payload is None:
                    continue
                chunk_df = txt_bytes_to_chunks_df(payload, path_str, params=params)
                if not chunk_df.empty:
                    out_dfs.append(chunk_df)
            except Exception:
                continue
```

- [ ] **Step 4: Run actor tests**

Run:

```bash
uv run --project nemo_retriever pytest nemo_retriever/tests/test_actor_operators.py::TestTxtSplitActor -q
```

Expected: PASS. If this environment lacks optional test dependencies, record the missing package and still run the syntax check in Step 5.

- [ ] **Step 5: Run syntax check**

Run:

```bash
python -m py_compile nemo_retriever/src/nemo_retriever/txt/ray_data.py nemo_retriever/tests/test_actor_operators.py
```

Expected: no output and exit code `0`.

- [ ] **Step 6: Commit binary payload patch**

Run:

```bash
git add nemo_retriever/src/nemo_retriever/txt/ray_data.py nemo_retriever/tests/test_actor_operators.py
git commit -m "fix: normalize txt ray data byte payloads"
```

### Task 3: Profile Template and Launcher

**Files:**
- Create: `tools/slurm-ray-smoke/cluster.example.env`
- Create: `tools/slurm-ray-smoke/submit.sh`

- [ ] **Step 1: Create the cluster profile template**

Create `tools/slurm-ray-smoke/cluster.example.env`:

```bash
# Copy this file to a file such as cluster.my-slurm.env and edit values for one SLURM login alias.

CLUSTER_ALIAS=slurm-login
SCRATCH_ROOT=/shared/scratch/${USER}/ray-xcluster

NODES=2
NTASKS_PER_NODE=1
CPUS_PER_TASK=8
TIME_LIMIT=00:30:00

RAY_PORT=6379
RAY_DASHBOARD_PORT=8265
RAY_CLIENT_SERVER_PORT=10001

SBATCH_SELECTOR=$(cat <<'SBATCH_SELECTOR_EOF'
#SBATCH --partition=debug
#SBATCH --account=research
SBATCH_SELECTOR_EOF
)
```

- [ ] **Step 2: Create the local launcher**

Create `tools/slurm-ray-smoke/submit.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'USAGE'
Usage: tools/slurm-ray-smoke/submit.sh path/to/cluster.my-slurm.env

The profile is a shell file containing CLUSTER_ALIAS, SCRATCH_ROOT, NODES,
NTASKS_PER_NODE, CPUS_PER_TASK, TIME_LIMIT, optional Ray ports, and
SBATCH_SELECTOR.
USAGE
}

[[ $# -eq 1 ]] || {
  usage
  exit 2
}

profile=$1
[[ -f "$profile" ]] || die "profile not found: $profile"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel)"
kit_dir="$script_dir"
template="$kit_dir/templates/job.sbatch.sh"
[[ -f "$template" ]] || die "missing template: $template"

known_profile_vars=(
  CLUSTER_ALIAS
  SCRATCH_ROOT
  SBATCH_SELECTOR
  NODES
  NTASKS_PER_NODE
  CPUS_PER_TASK
  TIME_LIMIT
  RAY_PORT
  RAY_DASHBOARD_PORT
  RAY_CLIENT_SERVER_PORT
  SOURCE_ROOT
)

while IFS='=' read -r name _value; do
  [[ "$name" =~ ^[A-Z][A-Z0-9_]*$ ]] || continue
  found=0
  for known in "${known_profile_vars[@]}"; do
    if [[ "$name" == "$known" ]]; then
      found=1
      break
    fi
  done
  [[ "$found" -eq 1 ]] || die "unsupported profile field: $name"
done < <(grep -E '^[A-Z][A-Z0-9_]*=' "$profile" || true)

# shellcheck source=/dev/null
source "$profile"

: "${CLUSTER_ALIAS:?profile must set CLUSTER_ALIAS}"
: "${SCRATCH_ROOT:?profile must set SCRATCH_ROOT}"
: "${NODES:?profile must set NODES}"
: "${NTASKS_PER_NODE:?profile must set NTASKS_PER_NODE}"
: "${CPUS_PER_TASK:?profile must set CPUS_PER_TASK}"
: "${TIME_LIMIT:?profile must set TIME_LIMIT}"

[[ "$CLUSTER_ALIAS" =~ ^[A-Za-z0-9_.-]+$ ]] || die "CLUSTER_ALIAS may contain only letters, numbers, dot, underscore, and dash"
[[ "$SCRATCH_ROOT" != *[[:space:]]* ]] || die "SCRATCH_ROOT must not contain whitespace"
[[ "$NODES" =~ ^[0-9]+$ ]] || die "NODES must be an integer"
[[ "$NTASKS_PER_NODE" =~ ^[0-9]+$ ]] || die "NTASKS_PER_NODE must be an integer"
[[ "$CPUS_PER_TASK" =~ ^[0-9]+$ ]] || die "CPUS_PER_TASK must be an integer"
[[ "$NODES" -ge 2 ]] || die "NODES must be at least 2 for this two-node smoke"
[[ "$NTASKS_PER_NODE" -ge 1 ]] || die "NTASKS_PER_NODE must be at least 1"
[[ "$CPUS_PER_TASK" -ge 2 ]] || die "CPUS_PER_TASK must be at least 2"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}"
SBATCH_SELECTOR="${SBATCH_SELECTOR:-}"

for port_name in RAY_PORT RAY_DASHBOARD_PORT RAY_CLIENT_SERVER_PORT; do
  port_value="${!port_name}"
  [[ "$port_value" =~ ^[0-9]+$ ]] || die "$port_name must be an integer"
done

source_root="${SOURCE_ROOT:-$repo_root/nemo_retriever}"
[[ -f "$source_root/pyproject.toml" ]] || die "nemo_retriever/pyproject.toml not found under source root: $source_root"
[[ -d "$kit_dir/smokes" ]] || die "missing smoke directory: $kit_dir/smokes"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-${CLUSTER_ALIAS}-nemo-intracluster-ray"
run_dir="${SCRATCH_ROOT%/}/nemo-intracluster/runs/${run_id}"

tmp_dir="$(mktemp -d)"
cleanup_tmp() {
  rm -rf "$tmp_dir"
}
trap cleanup_tmp EXIT

source_tar="$tmp_dir/nemo_retriever-src.tar.gz"
smokes_tar="$tmp_dir/smokes.tar.gz"
local_job="$tmp_dir/job.sbatch"

tar -C "$repo_root" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  -czf "$source_tar" \
  nemo_retriever

tar -C "$kit_dir" -czf "$smokes_tar" smokes

export RUN_DIR="$run_dir"
export NODES NTASKS_PER_NODE CPUS_PER_TASK TIME_LIMIT
export RAY_PORT RAY_DASHBOARD_PORT RAY_CLIENT_SERVER_PORT
export SBATCH_SELECTOR
export TEMPLATE_PATH="$template"
export OUTPUT_PATH="$local_job"

python3 - <<'PY'
from pathlib import Path
import os

template = Path(os.environ["TEMPLATE_PATH"]).read_text(encoding="utf-8")
selector = os.environ.get("SBATCH_SELECTOR", "").strip()
mapping = {
    "RUN_DIR": os.environ["RUN_DIR"],
    "NODES": os.environ["NODES"],
    "NTASKS_PER_NODE": os.environ["NTASKS_PER_NODE"],
    "CPUS_PER_TASK": os.environ["CPUS_PER_TASK"],
    "TIME_LIMIT": os.environ["TIME_LIMIT"],
    "RAY_PORT": os.environ["RAY_PORT"],
    "RAY_DASHBOARD_PORT": os.environ["RAY_DASHBOARD_PORT"],
    "RAY_CLIENT_SERVER_PORT": os.environ["RAY_CLIENT_SERVER_PORT"],
    "SBATCH_SELECTOR": selector,
}

rendered = template
for key, value in mapping.items():
    rendered = rendered.replace(f"@@{key}@@", value)

if "@@" in rendered:
    raise SystemExit("unrendered template marker remains in generated sbatch")

Path(os.environ["OUTPUT_PATH"]).write_text(rendered, encoding="utf-8")
PY

printf 'Staging run directory on %s:%s\n' "$CLUSTER_ALIAS" "$run_dir"
ssh "$CLUSTER_ALIAS" "mkdir -p '$run_dir/source' '$run_dir/logs'"
scp "$source_tar" "$CLUSTER_ALIAS:$run_dir/source/nemo_retriever-src.tar.gz"
scp "$smokes_tar" "$CLUSTER_ALIAS:$run_dir/smokes.tar.gz"
scp "$local_job" "$CLUSTER_ALIAS:$run_dir/job.sbatch"

job_id="$(
  ssh "$CLUSTER_ALIAS" "cd '$run_dir' && tar -xzf smokes.tar.gz && sbatch --parsable job.sbatch"
)"

cat <<EOF
cluster=$CLUSTER_ALIAS
run_id=$run_id
run_dir=$run_dir
job_id=$job_id
EOF
```

- [ ] **Step 3: Make launcher executable and shell-parse it**

Run:

```bash
chmod +x tools/slurm-ray-smoke/submit.sh
bash -n tools/slurm-ray-smoke/submit.sh
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Commit launcher files**

Run:

```bash
git add tools/slurm-ray-smoke/cluster.example.env tools/slurm-ray-smoke/submit.sh
git commit -m "feat: add slurm ray smoke launcher"
```

### Task 4: Generated sbatch Template

**Files:**
- Create: `tools/slurm-ray-smoke/templates/job.sbatch.sh`

- [ ] **Step 1: Create the sbatch template**

Create `tools/slurm-ray-smoke/templates/job.sbatch.sh`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=nemo-ray-smoke
#SBATCH --nodes=@@NODES@@
#SBATCH --ntasks-per-node=@@NTASKS_PER_NODE@@
#SBATCH --cpus-per-task=@@CPUS_PER_TASK@@
#SBATCH --time=@@TIME_LIMIT@@
#SBATCH --output=@@RUN_DIR@@/logs/slurm-%j.out
#SBATCH --error=@@RUN_DIR@@/logs/slurm-%j.err
@@SBATCH_SELECTOR@@

set -euo pipefail

RUN_DIR="@@RUN_DIR@@"
RAY_PORT="@@RAY_PORT@@"
RAY_DASHBOARD_PORT="@@RAY_DASHBOARD_PORT@@"
RAY_CLIENT_SERVER_PORT="@@RAY_CLIENT_SERVER_PORT@@"
PYTHON_VERSION="3.12"
RAY_VERSION="2.55.1"

LOG_DIR="$RUN_DIR/logs"
WORK_ROOT="/tmp/${USER}/nemo-ray-${SLURM_JOB_ID}"
UV_INSTALL_DIR="$WORK_ROOT/uv-bin"
UV_BIN="$UV_INSTALL_DIR/uv"
VENV_DIR="$WORK_ROOT/venv"
SOURCE_ROOT="$WORK_ROOT/source"
SRC_DIR="$SOURCE_ROOT/nemo_retriever"
SOURCE_TARBALL="$RUN_DIR/source/nemo_retriever-src.tar.gz"
INPUT_DIR="$RUN_DIR/input"
TOKENIZER_DIR="$RUN_DIR/tokenizer"

export RUN_DIR LOG_DIR WORK_ROOT UV_INSTALL_DIR UV_BIN VENV_DIR SOURCE_ROOT SRC_DIR SOURCE_TARBALL
export PYTHON_VERSION RAY_VERSION INPUT_DIR TOKENIZER_DIR
export RETRIEVER_BUILD_DATE=20260508
export RETRIEVER_BUILD_NUMBER=0

mkdir -p "$LOG_DIR" "$INPUT_DIR" "$TOKENIZER_DIR"

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${ALLOC_NODES[0]}"
WORKER_NODES=("${ALLOC_NODES[@]:1}")

HEAD_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export HEAD_NODE HEAD_IP RAY_ADDRESS

write_summary() {
  local status=$1
  {
    printf 'status=%s\n' "$status"
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'head_node=%s\n' "$HEAD_NODE"
    printf 'head_ip=%s\n' "$HEAD_IP"
    printf 'ray_address=%s\n' "$RAY_ADDRESS"
    printf 'nodes=%s\n' "${ALLOC_NODES[*]}"
  } > "$RUN_DIR/summary.env"
}

cleanup() {
  set +e
  {
    printf '[cleanup] stopping Ray on allocated nodes\n'
    for node in "${ALLOC_NODES[@]}"; do
      srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source '$VENV_DIR/bin/activate' && ray stop --force" || true
    done
  } >> "$LOG_DIR/ray-stop.log" 2>&1
}
trap cleanup EXIT
trap 'write_summary failed' ERR

cat > "$RUN_DIR/setup_node.sh" <<'SETUP_NODE'
#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$WORK_ROOT" "$UV_INSTALL_DIR" "$SOURCE_ROOT"
rm -rf "$SRC_DIR"
tar -xzf "$SOURCE_TARBALL" -C "$SOURCE_ROOT"

if [[ ! -x "$UV_BIN" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | /usr/bin/env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi

"$UV_BIN" python install "$PYTHON_VERSION"
"$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
source "$VENV_DIR/bin/activate"

"$UV_BIN" pip install --upgrade pip
"$UV_BIN" pip install "ray[data,serve]==${RAY_VERSION}" "transformers>=4.57.6,<5" "tokenizers>=0.20.3"
"$UV_BIN" pip install "$SRC_DIR"

python - <<'PY'
import nemo_retriever
print("nemo_retriever imported from", nemo_retriever.__file__)
PY
SETUP_NODE
chmod +x "$RUN_DIR/setup_node.sh"

printf '[setup] installing uv/python/source on %s nodes\n' "$SLURM_NNODES" | tee "$LOG_DIR/setup.log"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 "$RUN_DIR/setup_node.sh" >> "$LOG_DIR/setup.log" 2>&1

cat > "$RUN_DIR/ray_head.sh" <<'RAY_HEAD'
#!/usr/bin/env bash
set -euo pipefail
source "$VENV_DIR/bin/activate"
exec ray start \
  --head \
  --node-ip-address="$HEAD_IP" \
  --port="$RAY_PORT" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="$RAY_DASHBOARD_PORT" \
  --ray-client-server-port="$RAY_CLIENT_SERVER_PORT" \
  --num-cpus="$SLURM_CPUS_PER_TASK" \
  --resources='{"nemo_head": 1}' \
  --block
RAY_HEAD
chmod +x "$RUN_DIR/ray_head.sh"

cat > "$RUN_DIR/ray_worker.sh" <<'RAY_WORKER'
#!/usr/bin/env bash
set -euo pipefail
source "$VENV_DIR/bin/activate"
exec ray start \
  --address="$RAY_ADDRESS" \
  --num-cpus="$SLURM_CPUS_PER_TASK" \
  --resources='{"nemo_worker": 1}' \
  --block
RAY_WORKER
chmod +x "$RUN_DIR/ray_worker.sh"

printf '[ray] starting head on %s (%s)\n' "$HEAD_NODE" "$HEAD_IP" | tee "$LOG_DIR/ray-start.log"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" -w "$HEAD_NODE" "$RUN_DIR/ray_head.sh" > "$LOG_DIR/ray-head.log" 2>&1 &
ray_head_pid=$!

for worker_node in "${WORKER_NODES[@]}"; do
  printf '[ray] starting worker on %s\n' "$worker_node" | tee -a "$LOG_DIR/ray-start.log"
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" -w "$worker_node" "$RUN_DIR/ray_worker.sh" > "$LOG_DIR/ray-worker-${worker_node}.log" 2>&1 &
done

wait_for_ray() {
  for attempt in $(seq 1 120); do
    if "$VENV_DIR/bin/python" - <<'PY' >> "$LOG_DIR/ray-status.log" 2>&1; then
import os
import ray

expected_nodes = int(os.environ["SLURM_NNODES"])
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
alive = [node for node in ray.nodes() if node.get("Alive")]
resources = ray.cluster_resources()
assert len(alive) >= expected_nodes, (len(alive), expected_nodes)
assert resources.get("nemo_head", 0) >= 1, resources
assert resources.get("nemo_worker", 0) >= 1, resources
ray.shutdown()
PY
      return 0
    fi
    sleep 5
  done
  return 1
}

wait_for_ray
"$VENV_DIR/bin/ray" status --address "$RAY_ADDRESS" > "$LOG_DIR/ray-status-final.log" 2>&1

cat > "$RUN_DIR/run_smokes.sh" <<'RUN_SMOKES'
#!/usr/bin/env bash
set -euo pipefail

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$RUN_DIR/smokes:$SRC_DIR/src:${PYTHONPATH:-}"

mkdir -p "$INPUT_DIR" "$TOKENIZER_DIR"

cat > "$INPUT_DIR/doc-1.txt" <<'DOC'
alpha beta gamma delta epsilon zeta eta theta
DOC
cat > "$INPUT_DIR/doc-2.txt" <<'DOC'
iota kappa lambda mu nu xi omicron pi
DOC
cat > "$INPUT_DIR/doc-3.txt" <<'DOC'
rho sigma tau upsilon phi chi psi omega
DOC

python - <<'PY'
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import os

vocab = {"[UNK]": 0}
for word in "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split():
    vocab.setdefault(word, len(vocab))

tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
Path(os.environ["TOKENIZER_DIR"]).mkdir(parents=True, exist_ok=True)
fast.save_pretrained(os.environ["TOKENIZER_DIR"])
PY

python "$RUN_DIR/smokes/nemo_executor_smoke.py" \
  --ray-address "$RAY_ADDRESS" \
  --input-glob "$INPUT_DIR/*.txt" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-json "$RUN_DIR/nemo_executor_smoke.result.json"

python "$RUN_DIR/smokes/nemo_graph_ingestor_smoke.py" \
  --ray-address "$RAY_ADDRESS" \
  --input-glob "$INPUT_DIR/*.txt" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-json "$RUN_DIR/nemo_graph_ingestor_smoke.result.json"
RUN_SMOKES
chmod +x "$RUN_DIR/run_smokes.sh"

srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 -w "$HEAD_NODE" "$RUN_DIR/run_smokes.sh" > "$LOG_DIR/smokes.log" 2>&1

write_summary succeeded
printf '[done] smoke artifacts written to %s\n' "$RUN_DIR"
```

- [ ] **Step 2: Shell-parse the template**

Run:

```bash
bash -n tools/slurm-ray-smoke/templates/job.sbatch.sh
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Render with the example profile and shell-parse the generated job**

Run:

```bash
tmp_dir="$(mktemp -d)"
source tools/slurm-ray-smoke/cluster.example.env
RUN_DIR=/tmp/nemo-ray-smoke-render-test \
NODES="$NODES" \
NTASKS_PER_NODE="$NTASKS_PER_NODE" \
CPUS_PER_TASK="$CPUS_PER_TASK" \
TIME_LIMIT="$TIME_LIMIT" \
RAY_PORT="${RAY_PORT:-6379}" \
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}" \
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}" \
SBATCH_SELECTOR="$SBATCH_SELECTOR" \
TEMPLATE_PATH=tools/slurm-ray-smoke/templates/job.sbatch.sh \
OUTPUT_PATH="$tmp_dir/job.sbatch" \
python3 - <<'PY'
from pathlib import Path
import os

text = Path(os.environ["TEMPLATE_PATH"]).read_text(encoding="utf-8")
for key in ("RUN_DIR", "NODES", "NTASKS_PER_NODE", "CPUS_PER_TASK", "TIME_LIMIT", "RAY_PORT", "RAY_DASHBOARD_PORT", "RAY_CLIENT_SERVER_PORT", "SBATCH_SELECTOR"):
    text = text.replace(f"@@{key}@@", os.environ[key])
Path(os.environ["OUTPUT_PATH"]).write_text(text, encoding="utf-8")
PY
bash -n "$tmp_dir/job.sbatch"
rm -rf "$tmp_dir"
```

Expected: no output from `bash -n` and exit code `0`.

- [ ] **Step 4: Commit sbatch template**

Run:

```bash
git add tools/slurm-ray-smoke/templates/job.sbatch.sh
git commit -m "feat: generate slurm ray smoke jobs"
```

### Task 5: Smoke Scripts

**Files:**
- Create: `tools/slurm-ray-smoke/smokes/_common.py`
- Create: `tools/slurm-ray-smoke/smokes/nemo_executor_smoke.py`
- Create: `tools/slurm-ray-smoke/smokes/nemo_graph_ingestor_smoke.py`

- [ ] **Step 1: Add shared smoke helpers**

Create `tools/slurm-ray-smoke/smokes/_common.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_default(value: Any) -> str:
    return str(value)


def dataframe_sample(df: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.head(limit).to_dict(orient="records")


def write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def ray_state(ray_module: Any) -> dict[str, Any]:
    nodes = []
    for node in ray_module.nodes():
        nodes.append(
            {
                "node_id": node.get("NodeID"),
                "alive": bool(node.get("Alive")),
                "node_manager_address": node.get("NodeManagerAddress"),
                "resources": node.get("Resources", {}),
            }
        )
    return {
        "cluster_resources": ray_module.cluster_resources(),
        "available_resources": ray_module.available_resources(),
        "nodes": nodes,
    }


def alive_node_count(state: dict[str, Any]) -> int:
    return sum(1 for node in state["nodes"] if node["alive"])


def assert_cluster_ready(state: dict[str, Any], expected_nodes: int = 2) -> None:
    if alive_node_count(state) < expected_nodes:
        raise AssertionError(f"expected at least {expected_nodes} alive Ray nodes, got {alive_node_count(state)}")
    resources = state["cluster_resources"]
    for resource_name in ("nemo_head", "nemo_worker"):
        if resources.get(resource_name, 0) < 1:
            raise AssertionError(f"missing custom Ray resource {resource_name}: {resources}")
```

- [ ] **Step 2: Add direct RayDataExecutor smoke**

Create `tools/slurm-ray-smoke/smokes/nemo_executor_smoke.py`:

```python
from __future__ import annotations

import argparse
import socket
from typing import Any

import pandas as pd
import ray

from _common import assert_cluster_ready, dataframe_sample, ray_state, write_json
from nemo_retriever.graph import Graph, Node, RayDataExecutor, UDFOperator
from nemo_retriever.params import TextChunkParams
from nemo_retriever.txt.ray_data import TextChunkActor, TxtSplitActor


def _mark_host(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = socket.gethostname()
    return out


def mark_split_host(df: pd.DataFrame) -> pd.DataFrame:
    return _mark_host(df, "split_host")


def mark_chunk_host(df: pd.DataFrame) -> pd.DataFrame:
    return _mark_host(df, "chunk_host")


def _hosts(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df:
        return []
    return sorted(str(value) for value in df[column].dropna().unique())


def run(args: argparse.Namespace) -> dict[str, Any]:
    params = TextChunkParams(
        tokenizer_model_id=args.tokenizer_dir,
        max_tokens=6,
        overlap_tokens=0,
    )

    graph = Graph()
    graph.add_chain(
        Node(TxtSplitActor(params=params), name="TxtSplitActor"),
        Node(UDFOperator(mark_split_host, name="ProbeSplitHost"), name="ProbeSplitHost"),
        Node(TextChunkActor(params=params), name="TextChunkActor"),
        Node(UDFOperator(mark_chunk_host, name="ProbeChunkHost"), name="ProbeChunkHost"),
    )

    tiny_cpu = 0.25
    node_overrides = {
        "TxtSplitActor": {"resources": {"nemo_worker": 0.01}, "num_cpus": tiny_cpu, "concurrency": 1},
        "ProbeSplitHost": {"resources": {"nemo_worker": 0.01}, "num_cpus": tiny_cpu, "concurrency": 1},
        "TextChunkActor": {"resources": {"nemo_head": 0.01}, "num_cpus": tiny_cpu, "concurrency": 1},
        "ProbeChunkHost": {"resources": {"nemo_head": 0.01}, "num_cpus": tiny_cpu, "concurrency": 1},
    }

    executor = RayDataExecutor(
        graph,
        ray_address=args.ray_address,
        batch_size=1,
        num_cpus=tiny_cpu,
        num_gpus=0,
        node_overrides=node_overrides,
    )
    df = executor.ingest(args.input_glob)
    state = ray_state(ray)

    if not isinstance(df, pd.DataFrame):
        raise AssertionError(f"expected pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise AssertionError("direct executor smoke produced zero rows")

    assert_cluster_ready(state, expected_nodes=2)
    split_hosts = _hosts(df, "split_host")
    chunk_hosts = _hosts(df, "chunk_host")
    if not split_hosts:
        raise AssertionError("missing split_host annotations")
    if not chunk_hosts:
        raise AssertionError("missing chunk_host annotations")
    if split_hosts == chunk_hosts:
        raise AssertionError(f"expected split and chunk stages on different hosts, got {split_hosts}")

    return {
        "status": "succeeded",
        "ray_address": args.ray_address,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "split_hosts": split_hosts,
        "chunk_hosts": chunk_hosts,
        "ray": state,
        "sample": dataframe_sample(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    try:
        result = run(args)
    except Exception as exc:
        write_json(args.output_json, {"status": "failed", "error": repr(exc)})
        raise
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add GraphIngestor smoke**

Create `tools/slurm-ray-smoke/smokes/nemo_graph_ingestor_smoke.py`:

```python
from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import ray

from _common import assert_cluster_ready, dataframe_sample, ray_state, write_json
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import TextChunkParams


def run(args: argparse.Namespace) -> dict[str, Any]:
    params = TextChunkParams(
        tokenizer_model_id=args.tokenizer_dir,
        max_tokens=6,
        overlap_tokens=0,
    )

    node_overrides = {
        "MultiTypeExtractOperator": {
            "resources": {"nemo_worker": 0.01},
            "num_cpus": 0.25,
            "num_gpus": 0,
            "concurrency": 1,
        }
    }

    result = (
        GraphIngestor(
            run_mode="batch",
            documents=[args.input_glob],
            ray_address=args.ray_address,
            allow_no_gpu=True,
            batch_size=1,
            num_cpus=0.25,
            num_gpus=0,
            node_overrides=node_overrides,
        )
        .extract_txt(params)
        .ingest()
    )
    state = ray_state(ray)

    if not isinstance(result, pd.DataFrame):
        raise AssertionError(f"expected pandas DataFrame, got {type(result).__name__}")
    if result.empty:
        raise AssertionError("GraphIngestor smoke produced zero rows")

    assert_cluster_ready(state, expected_nodes=2)

    return {
        "status": "succeeded",
        "ray_address": args.ray_address,
        "rows": int(len(result)),
        "columns": list(result.columns),
        "ray": state,
        "sample": dataframe_sample(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    try:
        result = run(args)
    except Exception as exc:
        write_json(args.output_json, {"status": "failed", "error": repr(exc)})
        raise
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Compile smoke scripts**

Run:

```bash
python -m py_compile tools/slurm-ray-smoke/smokes/_common.py tools/slurm-ray-smoke/smokes/nemo_executor_smoke.py tools/slurm-ray-smoke/smokes/nemo_graph_ingestor_smoke.py
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Commit smoke scripts**

Run:

```bash
git add tools/slurm-ray-smoke/smokes
git commit -m "feat: add nemo retriever ray smoke scripts"
```

### Task 6: Handoff Documentation

**Files:**
- Create: `tools/slurm-ray-smoke/README.md`
- Create: `tools/slurm-ray-smoke/PATCH.md`
- Create: `tools/slurm-ray-smoke/triage.md`

- [ ] **Step 1: Add README quickstart**

Create `tools/slurm-ray-smoke/README.md`:

```markdown
# NeMo-Retriever SLURM Ray Smoke Kit

This kit submits a two-node SLURM job that starts Ray inside one cluster allocation and verifies NeMo-Retriever can use that external Ray cluster from both the direct graph executor and the `GraphIngestor` API.

## Prerequisites

- SSH access to a SLURM login alias.
- A shared scratch path visible from the login node and allocated compute nodes.
- Compute nodes can download uv, CPython 3.12, Ray, and tokenizer-only Python dependencies during the experiment.
- A NeMo-Retriever source checkout containing the temporary text tokenizer and byte payload patches described in `PATCH.md`.

## Quickstart

```bash
cd /path/to/NeMo-Retriever
cp tools/slurm-ray-smoke/cluster.example.env tools/slurm-ray-smoke/cluster.my-slurm.env
${EDITOR:-vi} tools/slurm-ray-smoke/cluster.my-slurm.env
tools/slurm-ray-smoke/submit.sh tools/slurm-ray-smoke/cluster.my-slurm.env
```

The submit command prints:

```text
cluster=slurm-login
run_id=20260508T153353Z-slurm-login-nemo-intracluster-ray
run_dir=/shared/scratch/user/ray-xcluster/nemo-intracluster/runs/20260508T153353Z-slurm-login-nemo-intracluster-ray
job_id=123456
```

## Profile Fields

- `CLUSTER_ALIAS`: SSH alias for the SLURM login node.
- `SCRATCH_ROOT`: shared scratch root where run directories are created.
- `SBATCH_SELECTOR`: multiline `#SBATCH` lines for partition, account, QOS, constraints, or GRES.
- `NODES`: node count. The first supported smoke uses `2`.
- `NTASKS_PER_NODE`: task count per node. Use `1`.
- `CPUS_PER_TASK`: CPU cores reserved per Ray process.
- `TIME_LIMIT`: SLURM wall time such as `00:30:00`.
- `RAY_PORT`, `RAY_DASHBOARD_PORT`, `RAY_CLIENT_SERVER_PORT`: port block used inside the allocation.

The launcher generates `job.sbatch`. If a normal run requires editing generated sbatch by hand, add a profile field first.

## Watching the Job

```bash
ssh slurm-login 'squeue -j 123456'
ssh slurm-login 'sacct -j 123456 --format=JobID,State,ExitCode,Elapsed,NodeList'
```

## Successful Artifacts

On success, inspect these files under `run_dir`:

- `summary.env`: contains `status=succeeded`, `head_node`, `head_ip`, `ray_address`, and allocated nodes.
- `nemo_executor_smoke.result.json`: direct `RayDataExecutor` proof with row count, live Ray nodes, `nemo_head`, `nemo_worker`, and different split/chunk host lists.
- `nemo_graph_ingestor_smoke.result.json`: `GraphIngestor(run_mode="batch", ray_address=...)` proof with row count, live Ray nodes, `nemo_head`, and `nemo_worker`.
- `logs/`: setup, Ray start/status/stop, smoke, and SLURM stdout/stderr logs.

## Known Baseline

The fixed ComputeLab run completed as SLURM job `2090092` on nodes `ipp1-1211` and `ipp1-1212`. Its direct smoke produced three rows with split work on `ipp1-1212`, chunk work on `ipp1-1211`, and both custom resources present.
```

- [ ] **Step 2: Add patch note**

Create `tools/slurm-ray-smoke/PATCH.md`:

```markdown
# Temporary Source Patch

This handoff currently needs two text-ingestion fixes in the NeMo-Retriever source checkout used by `submit.sh`.

## Local Tokenizer Paths

The smoke creates a tiny tokenizer in the run directory and passes that local path as `tokenizer_model_id`. `_get_tokenizer()` must skip `get_hf_revision()` when `model_id` already exists as a local path, then call `AutoTokenizer.from_pretrained(local_path, cache_dir=..., trust_remote_code=True)`.

## Ray/Pandas Byte Payloads

Ray Data can hand text split actors byte payloads as `bytes`, `bytearray`, `memoryview`, or Arrow-like scalar objects. `TxtSplitCPUActor.process()` must normalize these values before calling `txt_bytes_to_chunks_df()`.

## Removal Criteria

Keep this kit pointed at the SLURM-specific branch until both fixes land upstream or a released NeMo-Retriever version includes them. After that, remove this note from the quickstart and update `cluster.example.env` only if the released package changes runtime dependencies.
```

- [ ] **Step 3: Add triage guide**

Create `tools/slurm-ray-smoke/triage.md`:

```markdown
# SLURM Ray Smoke Triage

## Dynamic build version changes during install

Symptom: uv builds NeMo-Retriever twice and the dynamic version changes during the job.

First check:

```bash
grep -R "RETRIEVER_BUILD" logs/setup.log
```

Expected fix: `job.sbatch` exports stable `RETRIEVER_BUILD_DATE` and `RETRIEVER_BUILD_NUMBER` before package installation.

## Ray start succeeds but workers cannot connect

Symptom: `ray start` reports success in a short `srun`, but later `ray status` cannot see the worker.

First check:

```bash
grep -R "ray start" logs/ray-*.log
```

Expected fix: Ray head and worker run under blocking `ray start --block` commands launched by background `srun --overlap` steps.

## Wrong `env` binary on compute nodes

Symptom: uv installer or shebangs resolve a user-path `env` that is not valid on compute nodes.

First check:

```bash
grep -R "/usr/bin/env" job.sbatch logs/setup.log
```

Expected fix: installer commands use `/usr/bin/env` and generated scripts use absolute uv/venv paths for Python and Ray commands.

## Ray Data stalls on a tiny allocation

Symptom: Ray cluster is alive, but no smoke output appears after setup.

First check:

```bash
cat logs/ray-status-final.log
tail -n 100 logs/smokes.log
```

Expected fix: smoke overrides use low `num_cpus`, `concurrency=1`, and small custom resource fractions so the direct proof does not overreserve CPUs.

## Local tokenizer path fails revision lookup

Symptom: smoke logs show Hugging Face revision lookup for the run-local tokenizer path.

First check:

```bash
grep -R "get_hf_revision\|revision" logs/smokes.log
```

Expected fix: apply the local tokenizer path patch described in `PATCH.md`.

## Scratch path exists on login but not compute nodes

Symptom: setup fails to read `source/nemo_retriever-src.tar.gz` or write artifacts.

First check:

```bash
grep -R "nemo_retriever-src.tar.gz\|No such file" logs/setup.log logs/slurm-*.err
```

Expected fix: set `SCRATCH_ROOT` to a filesystem mounted on both the login node and allocated compute nodes.

## Queue remains pending on priority

Symptom: `squeue` shows the job pending and `sacct` has no terminal state.

First check:

```bash
JOB_ID=123456
squeue -j "$JOB_ID" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

Expected fix: wait for the scheduler or change `SBATCH_SELECTOR` to a partition/account/QOS that has available capacity.
```

- [ ] **Step 4: Commit docs**

Run:

```bash
git add tools/slurm-ray-smoke/README.md tools/slurm-ray-smoke/PATCH.md tools/slurm-ray-smoke/triage.md
git commit -m "docs: document slurm ray smoke handoff"
```

### Task 7: Local Verification

**Files:**
- Read: `tools/slurm-ray-smoke/submit.sh`
- Read: `tools/slurm-ray-smoke/templates/job.sbatch.sh`
- Read: `tools/slurm-ray-smoke/smokes/*.py`
- Read: `nemo_retriever/src/nemo_retriever/txt/split.py`
- Read: `nemo_retriever/src/nemo_retriever/txt/ray_data.py`

- [ ] **Step 1: Check shell scripts**

Run:

```bash
bash -n tools/slurm-ray-smoke/submit.sh
bash -n tools/slurm-ray-smoke/templates/job.sbatch.sh
```

Expected: no output and exit code `0` for both commands.

- [ ] **Step 2: Compile Python files**

Run:

```bash
python -m py_compile \
  nemo_retriever/src/nemo_retriever/txt/split.py \
  nemo_retriever/src/nemo_retriever/txt/ray_data.py \
  tools/slurm-ray-smoke/smokes/_common.py \
  tools/slurm-ray-smoke/smokes/nemo_executor_smoke.py \
  tools/slurm-ray-smoke/smokes/nemo_graph_ingestor_smoke.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run focused unit tests**

Run:

```bash
uv run --project nemo_retriever pytest \
  nemo_retriever/tests/test_txt_split.py::test_get_tokenizer_local_path_skips_revision_lookup \
  nemo_retriever/tests/test_actor_operators.py::TestTxtSplitActor::test_process_coerces_binary_payload_variants \
  -q
```

Expected: PASS. If the local environment lacks test dependencies, record the missing package names and do not mark cluster validation as blocked.

- [ ] **Step 4: Verify generated sbatch has no unreplaced markers**

Run:

```bash
tmp_dir="$(mktemp -d)"
source tools/slurm-ray-smoke/cluster.example.env
RUN_DIR=/tmp/nemo-ray-smoke-render-test \
NODES="$NODES" \
NTASKS_PER_NODE="$NTASKS_PER_NODE" \
CPUS_PER_TASK="$CPUS_PER_TASK" \
TIME_LIMIT="$TIME_LIMIT" \
RAY_PORT="${RAY_PORT:-6379}" \
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}" \
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10001}" \
SBATCH_SELECTOR="$SBATCH_SELECTOR" \
TEMPLATE_PATH=tools/slurm-ray-smoke/templates/job.sbatch.sh \
OUTPUT_PATH="$tmp_dir/job.sbatch" \
python3 - <<'PY'
from pathlib import Path
import os

text = Path(os.environ["TEMPLATE_PATH"]).read_text(encoding="utf-8")
for key in ("RUN_DIR", "NODES", "NTASKS_PER_NODE", "CPUS_PER_TASK", "TIME_LIMIT", "RAY_PORT", "RAY_DASHBOARD_PORT", "RAY_CLIENT_SERVER_PORT", "SBATCH_SELECTOR"):
    text = text.replace(f"@@{key}@@", os.environ[key])
Path(os.environ["OUTPUT_PATH"]).write_text(text, encoding="utf-8")
PY
! grep -R '@@' "$tmp_dir/job.sbatch"
bash -n "$tmp_dir/job.sbatch"
rm -rf "$tmp_dir"
```

Expected: `grep` finds no `@@` markers, `bash -n` exits `0`.

- [ ] **Step 5: Commit any local verification fixes**

If local verification required a fix, run:

```bash
git add tools/slurm-ray-smoke nemo_retriever/src/nemo_retriever/txt/split.py nemo_retriever/src/nemo_retriever/txt/ray_data.py nemo_retriever/tests/test_txt_split.py nemo_retriever/tests/test_actor_operators.py
git commit -m "fix: polish slurm ray smoke kit"
```

If local verification passed without fixes, skip this commit step.

### Task 8: ComputeLab Cluster Verification

**Files:**
- Read: `tools/slurm-ray-smoke/README.md`
- Read: generated remote `summary.env`
- Read: generated remote `nemo_executor_smoke.result.json`
- Read: generated remote `nemo_graph_ingestor_smoke.result.json`

- [ ] **Step 1: Create a ComputeLab profile from the example**

Run:

```bash
cp tools/slurm-ray-smoke/cluster.example.env tools/slurm-ray-smoke/cluster.computelab-sc-01.env
${EDITOR:-vi} tools/slurm-ray-smoke/cluster.computelab-sc-01.env
```

Expected profile values:

```bash
CLUSTER_ALIAS=computelab-sc-01
SCRATCH_ROOT=/home/scratch.${USER}/ray-xcluster
NODES=2
NTASKS_PER_NODE=1
CPUS_PER_TASK=8
TIME_LIMIT=00:30:00
```

Use the partition/account/QOS lines required by the ComputeLab login alias in `SBATCH_SELECTOR`.

- [ ] **Step 2: Submit the job**

Run:

```bash
tools/slurm-ray-smoke/submit.sh tools/slurm-ray-smoke/cluster.computelab-sc-01.env
```

Expected: the command prints `cluster=computelab-sc-01`, a timestamped `run_dir`, and a numeric `job_id`.

- [ ] **Step 3: Watch SLURM state**

Set `JOB_ID` from the submit output, then run:

```bash
JOB_ID=123456
ssh computelab-sc-01 "squeue -j '$JOB_ID' -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'"
ssh computelab-sc-01 "sacct -j '$JOB_ID' --format=JobID,State,ExitCode,Elapsed,NodeList"
```

Expected terminal state: `COMPLETED` and `ExitCode` equal to `0:0`.

- [ ] **Step 4: Validate artifacts**

Set `RUN_DIR` from the submit output, then run:

```bash
RUN_DIR=/home/scratch.user/ray-xcluster/nemo-intracluster/runs/20260508T153353Z-computelab-sc-01-nemo-intracluster-ray
ssh computelab-sc-01 "cat '$RUN_DIR/summary.env'"
ssh computelab-sc-01 "RUN_DIR='$RUN_DIR' python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR'])
for name in ('nemo_executor_smoke.result.json', 'nemo_graph_ingestor_smoke.result.json'):
    payload = json.loads((run_dir / name).read_text())
    assert payload['status'] == 'succeeded', payload
    assert payload['rows'] > 0, payload
    resources = payload['ray']['cluster_resources']
    assert resources.get('nemo_head', 0) >= 1, resources
    assert resources.get('nemo_worker', 0) >= 1, resources
    alive = [node for node in payload['ray']['nodes'] if node['alive']]
    assert len(alive) >= 2, payload['ray']['nodes']

direct = json.loads((run_dir / 'nemo_executor_smoke.result.json').read_text())
assert direct['split_hosts'] != direct['chunk_hosts'], direct
print('artifacts ok')
PY"
```

Expected: `summary.env` contains `status=succeeded`, and the Python validation prints `artifacts ok`.

- [ ] **Step 5: Commit ComputeLab profile decision**

Do not commit `cluster.computelab-sc-01.env` because it may contain account, partition, or scratch details. If README wording needed a correction after the run, commit only that documentation change:

```bash
git add tools/slurm-ray-smoke/README.md tools/slurm-ray-smoke/triage.md
git commit -m "docs: refine slurm smoke validation notes"
```

If the README and triage guide were already accurate, skip this commit step.

### Task 9: DLCluster Readiness Check

**Files:**
- Read: `tools/slurm-ray-smoke/README.md`
- Read: remote SLURM queue and artifact files when the job starts

- [ ] **Step 1: Create a DLCluster profile from the example**

Run:

```bash
cp tools/slurm-ray-smoke/cluster.example.env tools/slurm-ray-smoke/cluster.dlcluster.env
${EDITOR:-vi} tools/slurm-ray-smoke/cluster.dlcluster.env
```

Expected profile shape:

```bash
CLUSTER_ALIAS=dlcluster
SCRATCH_ROOT=/home/scratch.${USER}/ray-xcluster
NODES=2
NTASKS_PER_NODE=1
CPUS_PER_TASK=8
TIME_LIMIT=00:30:00
```

Use the partition/account/QOS/GRES lines required by DLCluster in `SBATCH_SELECTOR`.

- [ ] **Step 2: Submit the job**

Run:

```bash
tools/slurm-ray-smoke/submit.sh tools/slurm-ray-smoke/cluster.dlcluster.env
```

Expected: the command prints `cluster=dlcluster`, a timestamped `run_dir`, and a numeric `job_id`.

- [ ] **Step 3: Check scheduler state**

Set `JOB_ID` from the submit output, then run:

```bash
JOB_ID=123456
ssh dlcluster "squeue -j '$JOB_ID' -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'"
ssh dlcluster "sacct -j '$JOB_ID' --format=JobID,State,ExitCode,Elapsed,NodeList"
```

Expected: either `COMPLETED` with `ExitCode` equal to `0:0`, or `PENDING` with a scheduler reason such as `Priority`.

- [ ] **Step 4: Validate artifacts when DLCluster completes**

Run the same artifact validation command from Task 8 Step 4 with `ssh dlcluster` and the DLCluster `RUN_DIR`.

Expected after completion: `summary.env` contains `status=succeeded`, and both result JSON files pass the resource and row-count assertions.

- [ ] **Step 5: Record DLCluster status in the execution handoff**

If the job is still pending, state the exact pending reason and estimated start time in the final execution notes. If it completes, state the job ID, nodes, and artifact directory in the final execution notes. Do not commit `cluster.dlcluster.env`.

## Review Notes

- Spec coverage: the plan includes source patch tasks, generated sbatch flow, compact profile, staging launcher, direct `RayDataExecutor` smoke, `GraphIngestor` smoke, patch explanation, triage guide, local verification, ComputeLab verification, and DLCluster readiness.
- Red-flag scan: no deferred implementation phrases are present. The `@@NAME@@` tokens are real template markers consumed by `submit.sh`.
- Type consistency: the Python smoke scripts use `TextChunkParams`, `Graph`, `Node`, `RayDataExecutor`, `UDFOperator`, `TxtSplitActor`, `TextChunkActor`, and `GraphIngestor` names that exist in the current source tree. The launcher and sbatch template share the same rendered variables.
