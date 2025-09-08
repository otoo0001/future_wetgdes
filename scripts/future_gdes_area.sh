#!/bin/bash
# submit_wetgde_all.sh
# Run future_gdes_area.py for every (scenario × driver family) in a SLURM array.

set -euo pipefail

MAIL="nicholetylor@gmail.com"
PYTHON_SCRIPT="/home/otoo0001/github/paper_3/future_gdes/scripts/future_gdes_area.py"
ENV_SCRIPT="/home/otoo0001/load_all_default.sh"

# ── resources ────────────────────────────────────────────────────────────────
CORES=8
MEM="120G"
WALL="72:00:00"
PARTITION="genoa"

# ── IO, logs ─────────────────────────────────────────────────────────────────
OUT_DIR="/projects/prjs1578/futurewetgde/wetGDEs_area_scenarios"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# ── scenarios & driver families (cartesian product) ──────────────────────────
# COUNTERFACTUAL family values (match python): full | climate_only | landuse_only
SCENARIOS=("historical" "ssp126" "ssp370" "ssp585")
DRIVER_FAMILIES=("full" "climate_only" "landuse_only")

NUM_SCEN=${#SCENARIOS[@]}
NUM_FAM=${#DRIVER_FAMILIES[@]}
TOTAL=$(( NUM_SCEN * NUM_FAM ))
ARRAY_MAX=$(( TOTAL - 1 ))

# ── common inputs (export to job env) ────────────────────────────────────────
export MASK_DIR="/projects/prjs1578/futurewetgde/wetGDEs"
export BIOME_SHP="/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp"
export QA_DIR="/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/"
export OUT_DIR
export LOG_DIR

# performance knobs (avoid thread oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# batching and tiling
export TILE_Y=2048
export TILE_X=2048
export TIME_BATCH=24
export APPLY_QA_MASK=1

# engines and options
export XR_ENGINE=netcdf4
export WET_THRESHOLD=gt0          # gt0 | ge0.25 | ge0.5 | eq1

# quick smoke-test (set to 0 for full runs)
export SMALL_TEST=0

# LUH2 options
export APPLY_AG_MASK=1
export AG_RULE=exclude_when_ag    # or exclude_after_conversion
export AG_BASE_YEAR=2000
export LUH2_SSP_ROOT="/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/future_agric_area/ssp"

# NetCDF live streaming options
export WRITE_NC=1
export WRITE_NC_STREAMING=1
export NC_TRANSIENT=1
export NC_TMPDIR="/scratch-shared/${USER}"

# Parquet streaming options
export WRITE_PARQUET_STREAMING=1
export PARQUET_LIVE_DIR="${OUT_DIR}"
export WRITE_PARQUET_FINAL=1
export PARQUET_CODEC=snappy

# Optional preview map (small)
export PREVIEW_MAP=0
export PREVIEW_EVERY_N_MONTHS=12
export PREVIEW_EXCLUSION=crops     # none|crops|crops_pasture

# Pass arrays into the job script via strings
SCENARIOS_LIST="${SCENARIOS[*]}"
DRIVER_FAMILIES_LIST="${DRIVER_FAMILIES[*]}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=wetGDE_area_all
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=${MEM}
#SBATCH --array=0-${ARRAY_MAX}
#SBATCH --output=${LOG_DIR}/wetGDE_%A_%a.out
#SBATCH --error=${LOG_DIR}/wetGDE_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}
#SBATCH --export=ALL

set -euo pipefail

module purge
set +u
. "${ENV_SCRIPT}"                 # activate python env (modules + venv)
set -u

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${NC_TMPDIR}"

# reconstruct arrays inside the job
SCENARIOS=( ${SCENARIOS_LIST} )
DRIVER_FAMILIES=( ${DRIVER_FAMILIES_LIST} )
NUM_SCEN=${#SCENARIOS[@]}

# map array id -> (family_idx, scen_idx)
fam_idx=\$(( SLURM_ARRAY_TASK_ID / NUM_SCEN ))
scen_idx=\$(( SLURM_ARRAY_TASK_ID % NUM_SCEN ))

export SCENARIO="\${SCENARIOS[\$scen_idx]}"
export COUNTERFACTUAL="\${DRIVER_FAMILIES[\$fam_idx]}"

# Choose a RUN_TAG to prevent overwrites; use family name (and threshold)
export RUN_TAG="\${COUNTERFACTUAL}"

# Family-specific env for the python:
if [ "\${COUNTERFACTUAL}" = "climate_only" ]; then
  # Freeze ag to this year:
  export FIX_AG_YEAR=2000
fi
if [ "\${COUNTERFACTUAL}" = "landuse_only" ]; then
  # Freeze wet to historical monthly climatology from this file/window:
  export HIST_WET_FILE="\${MASK_DIR}/wetGDE_historical.nc"
  export HIST_CLIM_WINDOW="1985-01-01,2014-12-31"
fi

echo "JobID=\$SLURM_JOB_ID  TaskID=\$SLURM_ARRAY_TASK_ID  Node(s)=\$SLURM_JOB_NODELIST"
echo "Scenario=\$SCENARIO  Family(COUNTERFACTUAL)=\$COUNTERFACTUAL  RUN_TAG=\$RUN_TAG"
echo "SMALL_TEST=\$SMALL_TEST  TIME_BATCH=\$TIME_BATCH  WET_THRESHOLD=\$WET_THRESHOLD"
echo "WRITE_NC=\$WRITE_NC  STREAMING_NC=\$WRITE_NC_STREAMING  NC_TRANSIENT=\$NC_TRANSIENT  NC_TMPDIR=\$NC_TMPDIR"

ulimit -n 4096

python -u "${PYTHON_SCRIPT}"

# Move transient NC back (filename includes RUN_TAG)
if [ "\${NC_TRANSIENT}" = "1" ]; then
  src_nc="\${NC_TMPDIR}/gde_area_by_biome_realm_monthly_\${RUN_TAG}_\${SCENARIO}.nc"
  if [ -f "\$src_nc" ]; then
    mv -f "\$src_nc" "\${OUT_DIR}/"
    echo "Moved \$(basename "\$src_nc") to \${OUT_DIR}"
  fi
fi
EOF
