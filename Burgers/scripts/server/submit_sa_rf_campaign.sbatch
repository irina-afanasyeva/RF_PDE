#!/usr/bin/env bash
#SBATCH --job-name=sa_rf_one_seed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
# #SBATCH --partition=REPLACE_ME       # uncomment and fill in: this cluster's partition/queue name
# #SBATCH --account=REPLACE_ME         # uncomment and fill in: allocation/account, if required
# #SBATCH --gres=gpu:1                 # uncomment ONLY if a GPU is actually requested/needed (see
#                                       # resource-estimate note: this model is small and may not
#                                       # benefit much from a GPU -- CPU-only is a safe default)
#
# Slurm template for the SA-RF one-seed controlled comparison, intended
# for hpc-portal2.hpc.uark.edu or a similar Slurm-managed cluster.
# NOT YET SUBMITTED -- fill in the placeholders above (partition/account),
# verify the host-key situation for hpc-portal2.hpc.uark.edu yourself
# before connecting (this session's SSH attempt found a HOST KEY
# MISMATCH -- do not bypass this warning without independently verifying
# the new key through an official/out-of-band channel), then submit with:
#   sbatch scripts/server/submit_sa_rf_campaign.sbatch
set -euo pipefail

REPO_DIR="${REPO_DIR:-/path/to/RF_PDE}"   # EDIT: absolute path to the repo clone on the cluster
CONDA_ENV="${CONDA_ENV:-}"                 # EDIT: e.g. "rf_pde" -- leave empty to use module-loaded python3

cd "${REPO_DIR}/Burgers"

if [ -n "${CONDA_ENV}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-n/a}"
python3 -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

python3 -u scripts/run_sa_rf_one_seed_comparison.py
