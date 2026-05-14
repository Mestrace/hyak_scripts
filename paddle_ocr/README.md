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
Build to `/tmp` for speed, then move to your project directory, and you can build either

```bash
module load apptainer

# Use Either: 1) CPU Image
apptainer build --fakeroot /tmp/paddle_cpu.sif hyak_scripts/paddle_ocr/paddle_cpu.def
mv /tmp/paddle_cpu.sif .

# OR: 2) GPU Image
apptainer build --fakeroot /tmp/paddle_gpu.sif hyak_scripts/paddle_ocr/paddle_gpu.def
mv /tmp/paddle_gpu.sif .
```

## Step 2: Submitting Jobs

The `paddle_ocr.slurm` script is parameterized to handle both CPU and GPU workflows via environment variables passed through `--export`.

### Key Configuration Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INPUT`     | Path to directory of images OR a PDF file | (Required) |
| `OUTPUT_DIR`| Path to save OCR results | (Required) |
| `USE_GPU`   | Set to `true` to use GPU container and hardware | `false` |
| `ENABLE_HPI`| Enable High Performance Inference (HPI) | `true` |
| `WORKERS`   | Number of parallel threads for processing images | `1` |
| `CPU_THREADS`| Number of threads per Paddle instance | `$SLURM_CPUS_PER_TASK` |

Logs (stdout/stderr) will be saved in `/mmfs1/gscratch/stf/<your_netid>/log/`.

### Step 2.1: CPU Execution (Single File/Folder)
To run on CPU with MKLDNN and HPI optimization:
```bash
# For a directory of images:
sbatch --partition=ckpt-all --cpus-per-task=20 --mem=32g --export=ALL,INPUT=./images,OUTPUT_DIR=./output,WORKERS=2,CPU_THREADS=10,ENABLE_HPI=true paddle_ocr.slurm
```

### Step 2.2: GPU Execution
Set `USE_GPU=true`. HPI can still be enabled for supported layers:
```bash
sbatch --partition=ckpt-all --gres=gpu:1 --cpus-per-task=8 --mem=32g --export=ALL,INPUT=.,OUTPUT_DIR=./out/,WORKERS=1,CPU_THREADS=16,USE_GPU=true paddle_ocr.slurm
```

### Step 2.3: Job Array Execution (Multiple PDF Files)
To process many PDF files at once, use `paddle_ocr_pdf_array.slurm`.

1. **Create a task list** (a text file with one PDF path per line):
   ```bash
   find /path/to/pdfs -name "*.pdf" > pdf_list.txt
   ```

2. **Submit the array job**:
   ```bash
   # If pdf_list.txt has 50 files, use --array=0-49
   sbatch --partition=ckpt-all --array=0-49 --cpus-per-task=8 --mem=32g --export=ALL,INPUT_LIST=pdf_list.txt,OUTPUT_ROOT=./ocr_results paddle_ocr_pdf_array.slurm
   ```
   Each PDF will be processed in its own subdirectory within `OUTPUT_ROOT` (e.g., `./ocr_results/filename/`).

## Performance Tips: Single Job vs. Job Array

| Scenario | Recommended Approach | Reason |
|----------|----------------------|--------|
| **100-500 Images in one folder** | **Single Job** (`paddle_ocr.slurm`) | Avoids "Startup Tax" (loading models/containers 100x). Uses internal multi-threading (`WORKERS`) for speed. |
| **Dozens of PDF files** | **Job Array** (`paddle_ocr_pdf_array.slurm`) | Distributes heavy PDF decomposition and processing across many nodes simultaneously. |
| **10,000+ Images** | **Job Array** | Prevents a single job from timing out and allows for massive horizontal scaling. |

**Hint:** For most "archival box" workflows where you have a folder of JPEGs, a single job with high `--cpus-per-task` (e.g., 40) and `WORKERS=20` is the fastest path.
