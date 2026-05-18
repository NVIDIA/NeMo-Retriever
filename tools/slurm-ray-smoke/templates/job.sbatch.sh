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
export RAY_PORT RAY_DASHBOARD_PORT RAY_CLIENT_SERVER_PORT PYTHON_VERSION RAY_VERSION INPUT_DIR TOKENIZER_DIR
export RETRIEVER_BUILD_DATE=20260508
export RETRIEVER_BUILD_NUMBER=0

mkdir -p "$LOG_DIR" "$INPUT_DIR" "$TOKENIZER_DIR"

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${ALLOC_NODES[0]}"
WORKER_NODES=("${ALLOC_NODES[@]:1}")

HEAD_IP="$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')"
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

wait_for_ray_head() {
  for attempt in $(seq 1 120); do
    if "$VENV_DIR/bin/python" - <<'PY' >> "$LOG_DIR/ray-head-ready.log" 2>&1; then
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
resources = ray.cluster_resources()
assert resources.get("nemo_head", 0) >= 1, resources
ray.shutdown()
PY
      return 0
    fi
    sleep 5
  done
  return 1
}

wait_for_ray_head

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
assert resources.get("nemo_worker", 0) >= max(1, expected_nodes - 1), resources
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
alpha beta gamma delta epsilon zeta
DOC
cat > "$INPUT_DIR/doc-2.txt" <<'DOC'
iota kappa lambda mu nu xi
DOC
cat > "$INPUT_DIR/doc-3.txt" <<'DOC'
rho sigma tau upsilon phi chi
DOC

python - <<'PY'
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import os

vocab = {"[UNK]": 0}
for word in "alpha beta gamma delta epsilon zeta iota kappa lambda mu nu xi rho sigma tau upsilon phi chi".split():
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
