from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from paddleocr import PPStructureV3


def setup_logging():
    logger = logging.getLogger('paddle_ocr')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )

    # Info/Debug to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    # Warning/Error to stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger


logger = setup_logging()


@dataclass
class OCRConfig:
    input_path: Path
    output_folder: Path
    max_workers: int = 4
    force_rerun: bool = False
    enable_hpi: bool = True
    cpu_threads: int = 4
    use_gpu: bool = False


@dataclass
class ProcessResult:
    image_path: Path
    markdown_text: str
    success: bool


class FolderOCRProcessor:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        self._thread_local = threading.local()
        self._init_lock = threading.Lock()
        self._pipelines: list[PPStructureV3] = []

    def _get_pipeline(self):
        """
        Initialize and return a thread-local pipeline to ensure thread safety.
        """
        if not hasattr(self._thread_local, 'pipeline'):
            with self._init_lock:
                logger.info(
                    'Initializing PaddleOCR for '
                    f"thread: {threading.get_ident()}",
                )
                # Use PPStructureV3 from paddleocr
                pipeline = PPStructureV3(
                    lang='en',
                    use_doc_unwarping=True,
                    use_seal_recognition=False,
                    use_formula_recognition=False,
                    use_gpu=self.config.use_gpu,
                    enable_mkldnn=not self.config.use_gpu,
                    text_rec_score_thresh=0.15,
                    enable_hpi=self.config.enable_hpi,
                    cpu_threads=self.config.cpu_threads,
                )
                self._thread_local.pipeline = pipeline
                self._pipelines.append(pipeline)
        return self._thread_local.pipeline

    def _is_processed(self, image_path: Path) -> bool:
        if self.config.force_rerun:
            return False
        stem = image_path.stem
        md_file_path = self.config.output_folder / f"{stem}.md"
        return md_file_path.exists()

    def _process_image(self, image_path: Path) -> ProcessResult:
        stem = image_path.stem
        md_file_path = self.config.output_folder / f"{stem}.md"

        if self._is_processed(image_path):
            try:
                md_text = md_file_path.read_text(encoding='utf-8')
                return ProcessResult(image_path, md_text, True)
            except Exception:
                pass

        pipeline = self._get_pipeline()
        output = list(pipeline.predict(input=str(image_path)))

        if not output:
            return ProcessResult(image_path, '', False)

        res = output[0]
        res.save_to_img(save_path=str(self.config.output_folder))
        res.save_to_json(save_path=str(self.config.output_folder))
        res.save_to_markdown(save_path=str(self.config.output_folder))

        md_text = md_file_path.read_text(encoding='utf-8')
        return ProcessResult(image_path, md_text, True)

    def run(self) -> Path | None:
        """
        Process all images in the folder and yield a combined markdown file.
        """
        # Find all JPG/JPEG files
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG'):
            image_paths.extend([
                p for p in self.config.input_path.glob(ext)
                if (
                    not p.stem.endswith('_res') and
                    not p.stem.endswith('_preprocessed_img')
                )
            ])
        # Sort paths to ensure sequential markdown concatenation
        image_paths = sorted(image_paths)

        if not image_paths:
            logger.warning(f"No JPEG images found in {self.config.input_path}")
            return None
        try:
            results_text = []

            with ThreadPoolExecutor(
                max_workers=self.config.max_workers,
            ) as executor:
                results_iter = executor.map(self._process_image, image_paths)
                # tqdm removed for HPC batch compatibility
                for result in results_iter:
                    if result.success and result.markdown_text:
                        results_text.append(result.markdown_text)

            missing_mds = []
            for img_path in image_paths:
                expected_md = self.config.output_folder / f"{img_path.stem}.md"
                if not expected_md.exists():
                    missing_mds.append(expected_md.name)

            if missing_mds:
                raise FileNotFoundError(
                    f"Cannot combine markdowns. The following expected files "
                    f"are missing: {missing_mds}",
                )

            final_markdown = '\n\n'.join(results_text)

            combined_md_path = self.config.output_folder / \
                f"{self.config.input_path.name}_combined.md"
            combined_md_path.write_text(final_markdown, encoding='utf-8')
            return combined_md_path
        finally:
            for pipeline in self._pipelines:
                try:
                    pipeline.close()
                except Exception as e:
                    logger.error(f"Failed to close pipeline, exception: {e}")

            self._pipelines.clear()

            gc.collect()
            logger.info('All OCR pipelines closed and memory freed.')


def decompose_pdf(pdf_path: Path) -> Path:
    """
    Decompose each page of a PDF into an image and return the directory.
    Attempts to extract raw images if a page contains exactly one image object,
    otherwise falls back to high-resolution rendering.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError(
            'pypdfium2 is required for PDF processing. '
            "Please install it with 'pip install pypdfium2'",
        )

    output_dir = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Decomposing PDF: {pdf_path} -> {output_dir}")
    pdf = pdfium.PdfDocument(str(pdf_path))
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        image_path = output_dir / f"page_{i+1:03d}.jpg"

        # Try to extract raw image if it's a single-image page
        # We look for image objects specifically
        image_objects = [
            obj for obj in page.get_objects(
            ) if isinstance(obj, pdfium.PdfImage)
        ]
        extracted = False

        if len(image_objects) == 1:
            try:
                # Extract the raw bitmap of the image object
                bitmap = image_objects[0].get_bitmap()
                pil_image = bitmap.to_pil()
                pil_image.save(image_path, 'JPEG', quality=95)
                extracted = True
                logger.info(f"  Page {i+1}: Extracted raw image.")
            except Exception as e:
                logger.warning(
                    f"  Page {i+1}: Raw extraction failed ({e}), "
                    f"falling back to render.",
                )

        if not extracted:
            # Fallback to rendering the whole page at high resolution
            # (scale=6 ≈ 432 DPI)
            bitmap = page.render(scale=6)
            pil_image = bitmap.to_pil()
            pil_image.save(image_path, 'JPEG', quality=95)
            logger.info(f"  Page {i+1}: Rendered page at scale 6.")
        page.close()
    pdf.close()
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch OCR Process.')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument(
        '--cpu_threads', type=int,
        default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
    )
    parser.add_argument(
        '--enable_hpi', action='store_true',
        help='Enable High Performance Inference',
    )
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Use GPU for inference',
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        input_folder = input_path
    elif input_path.is_file() and input_path.suffix.lower() == '.pdf':
        input_folder = decompose_pdf(input_path)
    else:
        parser.error(
            f"Unsupported input type: {input_path}. "
            'Only directories or PDF files are allowed.',
        )

    config = OCRConfig(
        input_path=input_folder,
        output_folder=Path(args.output_dir),
        max_workers=args.workers,
        enable_hpi=args.enable_hpi,
        cpu_threads=args.cpu_threads,
        use_gpu=args.use_gpu,
    )

    processor = FolderOCRProcessor(config)
    final_output_path = processor.run()

    if final_output_path:
        logger.info(
            'Processing complete! Combined markdown saved at: '
            f"{final_output_path}",
        )
