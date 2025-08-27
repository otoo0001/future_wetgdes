#!/bin/bash
# submit_wetgde_all.sh
set -euo pipefail

MAIL="nicholetylor@gmail.com"
PYTHON_SCRIPT="/home/otoo0001/github/paper_3/future_gdes/scripts/future_gdes_area.py"
ENV_SCRIPT="/home/otoo0001/load_all_default.sh"

# resources
CORES=32
MEM="120G"
WALL="72:00:00"
PARTITION="genoa"

# io, logs
OUT_DIR="/projects/prjs1578/futurewetgde/wetGDEs_area"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# scenarios and array
SCENARIOS=("historical" "ssp126" "ssp370" "ssp585")
export MASK_DIR="/projects/prjs1578/futurewetgde/wetGDEs"
export BIOME_SHP="/gpfs/work2/0/prjs1578/futurewetgde/shapefiles/biomes_new/biomes/wwf_terr_ecos.shp"
export QA_DIR="/gpfs/work2/0/prjs1578/futurewetgde/quality_flags/"
export OUT_DIR
export LOG_DIR

# performance knobs, keep libraries single threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# new batching knobs, safe defaults
export TILE_Y=1024
export TILE_X=1024
export TIME_BATCH=6
export APPLY_QA_MASK=1

# engine hint
export XR_ENGINE=h5netcdf

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=wetgde_all
#SBATCH --partition=${PARTITION}
#SBATCH --time=${WALL}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --mem=${MEM}
#SBATCH --array=0-3
#SBATCH --output=${LOG_DIR}/wetGDE_%a.out
#SBATCH --error=${LOG_DIR}/wetGDE_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${MAIL}

set -euo pipefail
module purge
set +u
. "${ENV_SCRIPT}"
set -u

SCENARIOS=(${SCENARIOS[@]})
export SCENARIO="\${SCENARIOS[\$SLURM_ARRAY_TASK_ID]}"

python -u "${PYTHON_SCRIPT}"
EOF
