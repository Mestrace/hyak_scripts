#!/bin/bash
#SBATCH --job-name=paddle_ocr
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=.
#SBATCH --output=/mmfs1/gscratch/stf/%u/log/paddle_ocr_%j.out
#SBATCH --error=/mmfs1/gscratch/stf/%u/log/paddle_ocr_%j.err
#SBATCH --exclude=z[3005-3006] # no use of P100 (incompatible)

# Configuration - Update these or pass as environment variables
NETID=${NETID:-$(whoami)}
USE_GPU=${USE_GPU:-false}

# OCR Options
INPUT=${INPUT:-""}
OUTPUT_DIR=${OUTPUT_DIR:-""}
WORKERS=${WORKERS:-1}
CPU_THREADS=${CPU_THREADS:-$SLURM_CPUS_PER_TASK}
ENABLE_HPI=${ENABLE_HPI:-true}

# Absolute Paths
PROJECT_DIR="/mmfs1/gscratch/stf/${NETID}"
BASE_DIR="${PROJECT_DIR}/hyak_scripts/paddle_ocr"
LOG_DIR="${PROJECT_DIR}/log"
SCRIPT_PATH="${BASE_DIR}/main.py"

if [ "$USE_GPU" = "true" ]; then
    CONTAINER="${BASE_DIR}/paddle_gpu.sif"
    NV_FLAG="--nv"
    GPU_FLAG="--use_gpu"
    echo "Running in GPU mode..."
    export APPTAINERENV_LD_PRELOAD=/.singularity.d/libs/libcuda.so.1
else
    CONTAINER="${BASE_DIR}/paddle_cpu.sif"
    NV_FLAG=""
    GPU_FLAG=""
    echo "Running in CPU mode..."
    # --- MKLDNN Optimization ---
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Load Apptainer
module load apptainer

echo "Job ${SLURM_JOB_ID} starting at $(date)"

# Verification
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Apptainer image not found at ${CONTAINER}"
    exit 1
fi

if [ -z "$INPUT" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: INPUT and OUTPUT_DIR must be set."
    exit 1
fi

# HPI Flag logic
HPI_FLAG=""
if [ "$ENABLE_HPI" = "true" ]; then
    HPI_FLAG="--enable_hpi"
fi

# Run inside the container
echo "Running PaddleOCR..."
apptainer exec \
    ${NV_FLAG} \
    --home "${PROJECT_DIR}" \
    --bind /mmfs1/gscratch/stf \
    "${CONTAINER}" \
    python "${SCRIPT_PATH}" \
    --input "${INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    --cpu_threads "${CPU_THREADS}" \
    ${HPI_FLAG} \
    ${GPU_FLAG}

echo "Job completed at $(date)"
