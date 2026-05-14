#!/bin/bash
#SBATCH --job-name=paddle_ocr_array
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=.
#SBATCH --output=/mmfs1/gscratch/stf/%u/log/paddle_ocr_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/stf/%u/log/paddle_ocr_%A_%a.err
#SBATCH --exclude=z[3005-3006] # no use of P100 (incompatible)

# This script is designed to be used with a job array.
# Usage: sbatch --array=0-N --export=ALL,INPUT_LIST=tasks.txt,OUTPUT_ROOT=./results paddle_ocr_array.slurm
# Where tasks.txt contains one PDF path per line.

# Configuration - Update these or pass as environment variables
NETID=${NETID:-$(whoami)}
USE_GPU=${USE_GPU:-false}

# OCR Options
INPUT_LIST=${INPUT_LIST:-""}
OUTPUT_ROOT=${OUTPUT_ROOT:-""}
WORKERS=${WORKERS:-1}
CPU_THREADS=${CPU_THREADS:-$SLURM_CPUS_PER_TASK}
ENABLE_HPI=${ENABLE_HPI:-true}

# Absolute Paths
PROJECT_DIR="/mmfs1/gscratch/stf/${NETID}"
BASE_DIR="${PROJECT_DIR}/hyak_scripts/paddle_ocr"
LOG_DIR="${PROJECT_DIR}/log"
SCRIPT_PATH="${BASE_DIR}/main.py"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validation
if [ -z "$INPUT_LIST" ] || [ ! -f "$INPUT_LIST" ]; then
    echo "ERROR: INPUT_LIST must be set to a valid file path."
    exit 1
fi

if [ -z "$OUTPUT_ROOT" ]; then
    echo "ERROR: OUTPUT_ROOT must be set."
    exit 1
fi

# Get the specific PDF for this task index
# SLURM_ARRAY_TASK_ID is 0-indexed if submitted as 0-N
# We use sed to get the N+1 line because sed is 1-indexed
TASK_INDEX=$((SLURM_ARRAY_TASK_ID + 1))
INPUT_PDF=$(sed -n "${TASK_INDEX}p" "${INPUT_LIST}")

if [ -z "$INPUT_PDF" ]; then
    echo "ERROR: No input file found for task index ${SLURM_ARRAY_TASK_ID} in ${INPUT_LIST}"
    exit 1
fi

# Determine unique output directory for this PDF
PDF_NAME=$(basename "${INPUT_PDF}")
PDF_STEM="${PDF_NAME%.*}"
OUTPUT_DIR="${OUTPUT_ROOT}/${PDF_STEM}"

# Avoid duplicated efforts: check if output directory exists and is not empty
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; then
    echo "SKIPPING: ${OUTPUT_DIR} already exists and has content."
    echo "If you wish to re-process this PDF, please remove the directory and re-submit: rm -rf ${OUTPUT_DIR}"
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting at $(date)"
echo "Processing PDF: ${INPUT_PDF}"
echo "Output Directory: ${OUTPUT_DIR}"

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

# Load Apptainer
module load apptainer

# Verification
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Apptainer image not found at ${CONTAINER}"
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
    --input "${INPUT_PDF}" \
    --output_dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    --cpu_threads "${CPU_THREADS}" \
    ${HPI_FLAG} \
    ${GPU_FLAG}

echo "Job completed at $(date)"
