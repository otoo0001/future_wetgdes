#!/bin/bash
# submit_agg_and_plot_from_agg.sh
set -euo pipefail

MAIL="nicholetylor@gmail.com"

# === paths (edit these) ===
AGG_PY="/home/otoo0001/github/paper_3/future_gdes/scripts/future_gdes_area.py"
PLOT_PY="/home/otoo0001/github/paper_3/future_gdes/scripts/plot_future_nom.py"
ENV_SCRIPT="/home/otoo0001/load_pcrglobwb_python3_default.sh"

# === resources ===
PARTITION="genoa"
CORES=36
MEM="120G"
WALL="72:00:00"

# === IO ===
MASK_DIR="/projects/prjs1578/futurewetgde/wetGDEs"                         # source masks
OUT_DIR="/projects/prjs1578/futurewetgde/wetGDEs_area_test"                # aggregated outputs
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# scratch uses your username
SCRATCH="/scratch-shared/${USER}"
mkdir -p "${SCRATCH}" || true

# === scenarios ===
SCENARIOS=("historical" "ssp126" "ssp370" "ssp585")

# === optional cleanup to avoid duplicate streaming appends ===
# set to 1 to remove existing per-scenario files before recomputing
CLEAN_SCEN=0

submit_agg() {
  local scen="$1"
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=agg_${scen}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=${MEM}
#SBATCH --output=${LOG_DIR}/agg_${scen}.out
#SBATCH --error=${LOG_DIR}/agg_${scen}.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
umask 027

# env
if [ ! -f "${ENV_SCRIPT}" ]; then
  echo "ENV_SCRIPT not found: ${ENV_SCRIPT}" >&2; exit 1
fi
set +u; source "${ENV_SCRIPT}"; set -u

# threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# stability
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# aggregator config (matches your Python)
export MASK_DIR="${MASK_DIR}"
export OUT_DIR="${OUT_DIR}"
export LOG_DIR="${LOG_DIR}"
export QA_DIR="/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/"
export BIOME_SHP="/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp"

# LUH2
export APPLY_AG_MASK=1
export AG_RULE="exclude_when_ag"        # or exclude_after_conversion
export AG_BASE_YEAR=2000
export LUH2_SSP_ROOT="/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp"

# tiling / batching
export TILE_Y=2048
export TILE_X=2048
export TIME_BATCH=24
export APPLY_QA_MASK=1
export XR_ENGINE=netcdf4
export WET_THRESHOLD=gt0
export SMALL_TEST=0

# streaming outputs
export WRITE_NC=1
export WRITE_NC_STREAMING=1
export NC_TRANSIENT=0
export NC_TMPDIR="${SCRATCH}"
export WRITE_PARQUET_STREAMING=1
export PARQUET_LIVE_DIR="${OUT_DIR}"
export WRITE_PARQUET_FINAL=1
export PARQUET_CODEC=snappy

# scenario for this job
export SCENARIOS="${scen}"
export SCENARIO="${scen}"

# optional cleanup to avoid duplicating appends on re-run
if [ "${CLEAN_SCEN}" = "1" ]; then
  base="gde_area_by_biome_realm_monthly_${scen}"
  rm -f "${OUT_DIR}/gde_area_by_biome_realm_monthly_${scen}.nc" || true
  rm -rf "${OUT_DIR}/${base}" || true
  rm -f "${OUT_DIR}/gde_area_by_biome_realm_monthly_${scen}.parquet" || true
fi

python -u "${AGG_PY}"
EOF
}

submit_plot() {
  local after="$1"
  local plot_cores=8
  local plot_mem="32G"
  local plot_wall="24:00:00"

  sbatch --parsable --dependency=afterok:${after} <<EOF
#!/bin/bash
#SBATCH --job-name=plot_realms
#SBATCH --partition=${PARTITION}
#SBATCH --time=${plot_wall}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${plot_cores}
#SBATCH --mem=${plot_mem}
#SBATCH --output=${LOG_DIR}/plot_realms.out
#SBATCH --error=${LOG_DIR}/plot_realms.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
umask 027

# env
if [ ! -f "${ENV_SCRIPT}" ]; then
  echo "ENV_SCRIPT not found: ${ENV_SCRIPT}" >&2; exit 1
fi
set +u; source "${ENV_SCRIPT}"; set -u

# threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# plotting config (reads aggregator outputs)
export IN_DIR="${OUT_DIR}"
export PLOT_DIR="${OUT_DIR}/plots_from_agg"
export INPUT_KIND="auto"   # auto|parquet|netcdf
export SCENARIOS="historical,ssp126,ssp370,ssp585"
export EXCLUSIONS="none,crops,crops_pasture"
export SMOOTH_YEARS=5

python -u "${PLOT_PY}"
EOF
}

# 1) submit the per-scenario aggregation jobs
declare -a JIDS=()
for scen in "${SCENARIOS[@]}"; do
  jid=$(submit_agg "${scen}")
  echo "submitted agg ${scen}: ${jid}"
  JIDS+=("${jid}")
done

# 2) submit plotting job after all succeed
dep=$(IFS=: ; echo "${JIDS[*]}")
plot_jid=$(submit_plot "${dep}")
echo "submitted plotting job: ${plot_jid} after ${dep}"
