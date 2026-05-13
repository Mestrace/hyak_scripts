#!/bin/bash
#SBATCH --job-name=ocr_cleanup
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=.

# Configuration - Update these or pass as environment variables
NETID=${NETID:-$(whoami)}
# HF_MODEL should now be the absolute path to your .gguf file
HF_MODEL=${HF_MODEL:-"/mmfs1/gscratch/stf/${NETID}/models/qwen3-4b-instruct-2507-q8_0.gguf"}
DECK_TITLE=${DECK_TITLE:-"OCR Cleanup Project"}
SCAN_PATH=${SCAN_PATH:-"/path/to/your/scans"} # Update this!
PORT=8080

# Absolute Paths
PROJECT_DIR="/mmfs1/gscratch/stf/${NETID}"
BASE_DIR="${PROJECT_DIR}/hyak_scripts/ocr_cleanup"
LOG_DIR="${PROJECT_DIR}/log"
CONTAINER="${BASE_DIR}/ocr_cleanup.sif"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

echo "Job ${SLURM_JOB_ID} starting at $(date)"
echo "Project Directory: ${PROJECT_DIR}"
echo "Using Model Path: ${HF_MODEL}"

# Verification
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Apptainer image not found at ${CONTAINER}"
    echo "Please build it first: apptainer build --fakeroot --build-arg NETID=${NETID} ${CONTAINER} ${BASE_DIR}/ocr_cleanup.def"
    exit 1
fi

if [ ! -f "${HF_MODEL}" ]; then
    echo "ERROR: GGUF model file not found at ${HF_MODEL}"
    echo "Please download it first to your models folder."
    exit 1
fi

if [ ! -d "${SCAN_PATH}" ]; then
    echo "ERROR: Scan directory not found at ${SCAN_PATH}"
    exit 1
fi

echo "Starting llama-server..."

# 1. Start llama-server in the background
apptainer exec --bind /mmfs1/gscratch/stf --home "${PROJECT_DIR}" "${CONTAINER}" \
    llama-server -m "${HF_MODEL}" --port "${PORT}" --host 0.0.0.0 > "${LOG_DIR}/llama_server_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

# 2. Wait for server to be ready
echo "Waiting for llama-server to initialize..."
for i in {1..30}; do
    if curl -s "http://localhost:${PORT}/health" | grep -q '{"status":"ok"}'; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Server failed to start in time. Checking logs at ${LOG_DIR}/llama_server_${SLURM_JOB_ID}.log"
        kill $SERVER_PID
        exit 1
    fi
    sleep 10
done

# 3. Run the OCR cleanup script
echo "Running OCR cleanup script..."
apptainer exec --bind /mmfs1/gscratch/stf --home "${PROJECT_DIR}" "${CONTAINER}" \
    python3 "${BASE_DIR}/main.py" \
    --path "${SCAN_PATH}" \
    --deck-title "${DECK_TITLE}" \
    --base-url "http://localhost:${PORT}/v1"

# 4. Cleanup
echo "Cleaning up..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "Job completed at $(date)"
