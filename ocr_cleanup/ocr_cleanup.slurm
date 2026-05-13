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
# HF_MODEL should be the absolute path to your .gguf file
HF_MODEL=${HF_MODEL:-"/mmfs1/gscratch/stf/${NETID}/models/qwen3-4b-instruct-2507-q8_0.gguf"}
DECK_TITLE=${DECK_TITLE:-"OCR Cleanup Project"}
SCAN_PATH=${SCAN_PATH:-"/path/to/your/scans"} # Mandatory override
PORT=8080
CONCURRENCY=${CONCURRENCY:-2}
BASE_URL=${BASE_URL:-"http://localhost:${PORT}/v1"}

# Absolute Paths
PROJECT_DIR="/mmfs1/gscratch/stf/${NETID}"
BASE_DIR="${PROJECT_DIR}/hyak_scripts/ocr_cleanup"
LOG_DIR="${PROJECT_DIR}/log"
CONTAINER="${BASE_DIR}/ocr_cleanup.sif"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "Job ${SLURM_JOB_ID} starting at $(date)"
echo "Using Model Path: ${HF_MODEL}"
echo "Targeting API: ${BASE_URL} (Concurrency: ${CONCURRENCY})"

# Verification
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Apptainer image not found at ${CONTAINER}"
    exit 1
fi

if [[ "${BASE_URL}" == *"localhost"* ]] && [ ! -f "${HF_MODEL}" ]; then
    echo "ERROR: Local server requested but GGUF model missing at ${HF_MODEL}"
    exit 1
fi

if [ ! -d "${SCAN_PATH}" ]; then
    echo "ERROR: Scan directory not found at ${SCAN_PATH}"
    exit 1
fi

# 1. Start llama-server ONLY if targeting localhost
if [[ "${BASE_URL}" == *"localhost"* ]]; then
    # Calculate optimal threads: Total CPUs minus 2 for the Python script and OS overhead
    LLAMA_THREADS=$((SLURM_CPUS_PER_TASK - 2))
    if [ $LLAMA_THREADS -lt 1 ]; then LLAMA_THREADS=1; fi

    echo "Starting local llama-server with ${LLAMA_THREADS} threads..."
    apptainer exec --bind /mmfs1/gscratch/stf --home "${PROJECT_DIR}" "${CONTAINER}" \
        /app/llama-server -m "${HF_MODEL}" -t "${LLAMA_THREADS}" --port "${PORT}" --host 0.0.0.0 > "${LOG_DIR}/llama_server_${SLURM_JOB_ID}.log" 2>&1 &
    SERVER_PID=$!

    # 2. Wait for server readiness
    echo "Waiting for server to initialize..."
    for i in {1..30}; do
        if curl -s "http://localhost:${PORT}/health" | grep -q '{"status":"ok"}'; then
            echo "Server is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Server failed to start. Logs: ${LOG_DIR}/llama_server_${SLURM_JOB_ID}.log"
            kill $SERVER_PID
            exit 1
        fi
        sleep 10
    done
fi

# 3. Run the OCR cleanup script
echo "Running OCR cleanup script..."
apptainer exec --bind /mmfs1/gscratch/stf --home "${PROJECT_DIR}" "${CONTAINER}" \
    python3 "${BASE_DIR}/main.py" \
    --path "${SCAN_PATH}" \
    --deck-title "${DECK_TITLE}" \
    --base-url "${BASE_URL}" \
    --concurrency "${CONCURRENCY}"

# 4. Cleanup local server
if [ ! -z "${SERVER_PID}" ]; then
    echo "Shutting down local server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
fi

echo "Job completed at $(date)"
