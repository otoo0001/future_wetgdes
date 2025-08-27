#!/bin/bash
# submit_months_and_plot.sh
set -euo pipefail

MAIL="nicholetylor@gmail.com"
PYTHON_SCRIPT="/home/otoo0001/github/paper_3/future_gdes/scripts/plot_future_nom.py"

# resources
PARTITION="genoa"
CORES=36
MEM="120G"
WALL="72:00:00"

# IO
IN_DIR="/projects/prjs1578/futurewetgde/wetGDEs"
OUT_DIR="/projects/prjs1578/futurewetgde/wetGDEs"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# scenarios
SCENARIOS=("historical" "ssp126" "ssp370" "ssp585")

# env loader (make sure this file exists and activates your venv)
# e.g. it should do:
#   module purge
#   module load Python/<exact-version>
#   source ~/venvs/futuregde/bin/activate
ENV_SCRIPT="/home/otoo0001/load_pcrglobwb_python3_default.sh"

submit_compute() {
  local scen="$1"
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=months_${scen}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/months_${scen}.out
#SBATCH --error=${LOG_DIR}/months_${scen}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
umask 027

# source environment (fail fast if missing)
if [ ! -f "${ENV_SCRIPT}" ]; then
  echo "ENV_SCRIPT not found: ${ENV_SCRIPT}" >&2
  exit 1
fi
set +u; source "${ENV_SCRIPT}"; set -u

# threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_THREADPOOL_SIZE=${CORES}

# stability and IO
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# config to Python
export IN_DIR="${IN_DIR}"
export OUT_DIR="${OUT_DIR}"
export SPATIAL_CHUNK=1024
export PROGRESS_LOG="${LOG_DIR}/progress_${scen}.log"

# compute only this scenario's annual months
export SCENARIOS="${scen}"
export DO_PLOTS=0
export FORCE_REBUILD=0

python -u "${PYTHON_SCRIPT}"
EOF
}

submit_plot() {
  local after="$1"
  local plot_cores=8
  local plot_mem="32G"
  local plot_wall="24:00:00"

  sbatch --parsable --dependency=afterok:${after} <<EOF
#!/bin/bash
#SBATCH --job-name=months_plot_all
#SBATCH --partition=${PARTITION}
#SBATCH --time=${plot_wall}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${plot_cores}
#SBATCH --mem=${plot_mem}
#SBATCH --output=${LOG_DIR}/months_plot_all.out
#SBATCH --error=${LOG_DIR}/months_plot_all.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
umask 027

# source environment (fail fast if missing)
if [ ! -f "${ENV_SCRIPT}" ]; then
  echo "ENV_SCRIPT not found: ${ENV_SCRIPT}" >&2
  exit 1
fi
set +u; source "${ENV_SCRIPT}"; set -u

# threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DASK_THREADPOOL_SIZE=${plot_cores}

# stability and IO
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# config to Python
export IN_DIR="${IN_DIR}"
export OUT_DIR="${OUT_DIR}"
export SPATIAL_CHUNK=1024

# plot using all four scenarios; compute step will skip existing files
export SCENARIOS="historical ssp126 ssp370 ssp585"
export DO_PLOTS=1
export FORCE_REBUILD=0

python -u "${PYTHON_SCRIPT}"
EOF
}

# 1) submit compute jobs per scenario
declare -a JIDS=()
for scen in "${SCENARIOS[@]}"; do
  jid=$(submit_compute "${scen}")
  echo "submitted ${scen}: ${jid}"
  JIDS+=("${jid}")
done

# 2) submit plotting job after all compute jobs succeed
dep=$(IFS=: ; echo "${JIDS[*]}")
plot_jid=$(submit_plot "${dep}")
echo "submitted plotting job: ${plot_jid} after ${dep}"
