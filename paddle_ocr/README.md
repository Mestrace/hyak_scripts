# Running PaddlePaddle on Hyak (CPU & GPU)

This guide provides workflows for running PaddlePaddle on Hyak using both CPU-only nodes with MKLDNN (oneDNN) optimization and GPU nodes.

## Step 1: Setting Up Container Images

Building images should be done on a compute node to avoid straining the login nodes.

### Step 1.1: Request an Interactive Build Session
```bash
# Request a standard compute node for 1 hour
salloc -A stf -p compute-int --cpus-per-task=4 --mem=16G --time=1:00:00
```

### Step 1.2: Build the CPU/GPU Images
Build to `/tmp` for speed, then move to your project directory.

```bash
module load apptainer

# Use Either: 1) CPU Image
apptainer build --fakeroot /tmp/paddle_cpu.sif submodules/hyak_scripts/paddle_ocr/paddle_cpu.def
mv /tmp/paddle_cpu.sif submodules/hyak_scripts/paddle_ocr/

# OR: 2) GPU Image
apptainer build --fakeroot /tmp/paddle_gpu.sif submodules/hyak_scripts/paddle_ocr/paddle_gpu.def
mv /tmp/paddle_gpu.sif submodules/hyak_scripts/paddle_ocr/
```

## Step 2: Submitting Jobs

The `paddle_ocr.slurm` script is parameterized to handle both CPU and GPU workflows via environment variables passed through `--export`.

### Key Configuration Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INPUT`     | Path to directory of images OR a PDF file | (Required) |
| `OUTPUT_DIR`| Path to save OCR results | (Required) |
| `USE_GPU`   | Set to `true` to use GPU container and hardware | `false` |
| `ENABLE_HPI`| Enable High Performance Inference (HPI) | `false` |
| `WORKERS`   | Number of parallel threads for processing images | `1` |
| `CPU_THREADS`| Number of threads per Paddle instance | `$SLURM_CPUS_PER_TASK` |

Logs (stdout/stderr) will be saved in `/mmfs1/gscratch/stf/<your_netid>/log/`.

### Step 2.1: CPU Execution (Single File/Folder)
To run on CPU with MKLDNN and HPI optimization (HPI is recommended for CPU):
```bash
# For a directory of images:
sbatch --partition=ckpt-all --cpus-per-task=10 --mem=20g --export=ALL,INPUT=./images,OUTPUT_DIR=./output,WORKERS=2,CPU_THREADS=15,ENABLE_HPI=true paddle_ocr.slurm
```

### Step 2.2: GPU Execution
Set `USE_GPU=true`. **Note:** HPI should remain `false` for GPU as it is not compiled into the GPU image.
```bash
sbatch --partition=ckpt-all --gres=gpu:1 --cpus-per-task=2 --mem=8g --export=ALL,INPUT=.,OUTPUT_DIR=./out/,WORKERS=1,CPU_THREADS=8,USE_GPU=true paddle_ocr.slurm
```

> [!TIP]
> **Handling VRAM & Stability:**
> * **Use CPU for Stability:** If you have many large images or complex PDFs, set `USE_GPU=false`. CPU mode is the most stable as it uses system RAM (32GB+) rather than limited VRAM.
> * **RTX 2080 Ti (11GB):** These are great for standard images but may hit a `ResourceExhaustedError` (OOM) on very large files.
> * **Higher VRAM:** For heavy GPU workloads, request an A100 (`--gres=gpu:a100:1`) or L40.
> * **P100 Compatibility:** Do **not** use P100 GPUs (they require a different CUDA image); these are already excluded in the provided `.slurm` scripts.

### Step 2.3: Job Array Execution (Multiple PDF Files)
To process many PDF files at once, use `paddle_ocr_pdf_array.slurm`.

1. **Create a task list** (a text file with one PDF path per line):
   ```bash
   find /path/to/pdfs -name "*.pdf" > pdf_list.txt
   ```

2. **Submit the array job**:
   Use `%` to throttle (e.g., `--array=0-299%5`). Throttling is **CRITICAL** for large batches to avoid metadata bottlenecks on the `gscratch` filesystem.

   **CPU Array (Recommended for stability):**
   ```bash
   sbatch --partition=ckpt-all --array=0-299%5 --cpus-per-task=10 --mem=20g --export=ALL,INPUT_LIST=pdf_list.txt,OUTPUT_ROOT=./ocr_results,WORKERS=2,CPU_THREADS=15,ENABLE_HPI=true paddle_ocr_pdf_array.slurm
   ```

   **GPU Array:**
   ```bash
   sbatch --partition=ckpt-all --array=0-299%5 --gres=gpu:1 --cpus-per-task=2 --mem=8g --export=ALL,INPUT_LIST=pdf_list.txt,OUTPUT_ROOT=./ocr_results,WORKERS=1,CPU_THREADS=8,USE_GPU=true paddle_ocr_pdf_array.slurm
   ```
   Each PDF will be processed in its own subdirectory within `OUTPUT_ROOT` (e.g., `./ocr_results/filename/`).

## Performance Tips: Single Job vs. Job Array

| Scenario | Recommended Approach | Reason |
|----------|----------------------|--------|
| **100-500 Images in one folder** | **Single Job** (`paddle_ocr.slurm`) | Avoids "Startup Tax" (loading models/containers 100x). Uses internal multi-threading (`WORKERS`) for speed. |
| **Dozens of PDF files** | **Job Array** (`paddle_ocr_pdf_array.slurm`) | Distributes heavy PDF decomposition and processing across many nodes simultaneously. |
| **10,000+ Images** | **Job Array** | Prevents a single job from timing out and allows for massive horizontal scaling. |

**Hint:** For most "archival box" workflows where you have a folder of JPEGs, a single job with high `--cpus-per-task` (e.g., 40) and `WORKERS=20` is the fastest path.
