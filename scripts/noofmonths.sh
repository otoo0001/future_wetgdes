#!/bin/bash
# submit_wetgde_all.sh
set -euo pipefail

MAIL="nicholetylor@gmail.com"
PYTHON_SCRIPT="/home/otoo0001/github/paper_3/future_gdes/scripts/noofmonths_gdes.py"
ENV_SCRIPT="/home/otoo0001/load_all_default.sh"

CORES=36
MEM="120G"
WALL="72:00:00"
PARTITION="genoa"

# Logs (must match where the Python writes outputs)
OUT_DIR="/projects/prjs1578/futurewetgde/wetGDEs"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCENARIOS=("historical" "ssp126" "ssp370" "ssp585")

submit_one () {
  local scen="$1"
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=monthgde_${scen}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/wetGDE_${scen}.out
#SBATCH --error=${LOG_DIR}/wetGDE_${scen}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail

module purge
set +u
. "${ENV_SCRIPT}"
set -u

# threads and IO hygiene
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# knobs the Python honors
export SCENARIO="${scen}"
export PROGRESS_LOG="${LOG_DIR}/progress_${scen}.log"
export SPATIAL_CHUNK=1024
export DASK_THREADS=${CORES}

python -u "${PYTHON_SCRIPT}"
EOF
}

for scen in "${SCENARIOS[@]}"; do
  submit_one "${scen}"
done

