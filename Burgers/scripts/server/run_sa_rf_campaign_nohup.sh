#!/usr/bin/env bash
# Server-ready launcher for the SA-RF one-seed controlled comparison
# (baseline, adaptive, balanced_10, balanced_100), seed=0, coef_lr=0.001,
# 1000 Adam + 100 L-BFGS iterations ("controlled" budget in sa_rf/config.py).
#
# FOR A PLAIN SSH SERVER (no job scheduler), e.g. the GPU machine.
# NOT YET RUN -- inspect and edit the placeholders below, then launch
# yourself via: nohup bash scripts/server/run_sa_rf_campaign_nohup.sh &
#
# Uses `python3 -u` (unbuffered stdout) specifically to avoid the
# stdout-buffering issue discovered on the Mac run (empty log file after
# a SIGTERM because Python fully buffers redirected stdout) -- every
# print() is flushed immediately here.
set -euo pipefail

# --- EDIT THESE before running -------------------------------------------
REPO_DIR="${REPO_DIR:-/path/to/RF_PDE}"            # absolute path to the repo clone on the server
CONDA_ENV="${CONDA_ENV:-}"                           # e.g. "rf_pde" -- leave empty to use system python3
# ---------------------------------------------------------------------------

cd "${REPO_DIR}/Burgers"

if [ -n "${CONDA_ENV}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

LOG_DIR="outputs/sa_rf/one_seed_comparison_server_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
echo "Starting SA-RF one-seed comparison. Log: ${LOG_DIR}/run.log"

# Run the EXISTING, already-tested comparison script unmodified -- it
# already writes separate output directories per mode
# (outputs/sa_rf/one_seed_comparison/{baseline,adaptive,balanced_10,balanced_100}/
# config.json + metrics.csv + final.pt each) and a combined summary JSON.
nohup python3 -u scripts/run_sa_rf_one_seed_comparison.py \
    > "${LOG_DIR}/run.log" 2>&1 &

RUN_PID=$!
echo "${RUN_PID}" > "${LOG_DIR}/run.pid"
echo "Launched with PID ${RUN_PID}. Tail progress with: tail -f ${LOG_DIR}/run.log"
echo "Stop gracefully with: kill -TERM ${RUN_PID}"
