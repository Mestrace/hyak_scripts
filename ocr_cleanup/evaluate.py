from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path


def calculate_metrics(original: str, cleaned: str) -> dict:
    """Calculate various similarity and deviation metrics."""
    # Normalize for comparison (basic)
    orig_norm = re.sub(r'\s+', ' ', original).strip()
    clean_norm = re.sub(r'\s+', ' ', cleaned).strip()

    # 1. Sequence Similarity (Character level)
    # Higher is more similar.
    seq_matcher = difflib.SequenceMatcher(None, orig_norm, clean_norm)
    similarity = seq_matcher.ratio()

    # 2. Word Count Delta
    orig_words = orig_norm.split()
    clean_words = clean_norm.split()

    orig_count = len(orig_words)
    clean_count = len(clean_words)

    # Delta > 0 means the LLM added words (potential hallucination)
    # Delta < 0 means the LLM deleted content
    delta_words = clean_count - orig_count
    delta_percent = (delta_words / orig_count) if orig_count > 0 else 0

    # 3. Character Jaccard Similarity (Set of unique chars)
    # Helps detect if the LLM started using a completely different vocabulary
    orig_chars = set(orig_norm)
    clean_chars = set(clean_norm)
    intersection = orig_chars.intersection(clean_chars)
    union = orig_chars.union(clean_chars)
    jaccard = len(intersection) / len(union) if union else 1.0

    # 4. Deviation Flags
    flags = []
    if similarity < 0.85:
        flags.append('LOW_SIMILARITY')
    if delta_percent > 0.15:
        flags.append('POTENTIAL_HALLUCINATION (Significant Expansion)')
    if delta_percent < -0.20:
        flags.append('POTENTIAL_LOSS (Significant Deletion)')

    return {
        'similarity': round(similarity, 4),
        'orig_len': orig_count,
        'clean_len': clean_count,
        'delta_percent': round(delta_percent, 4),
        'jaccard': round(jaccard, 4),
        'flags': flags,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OCR cleanup quality.',
    )
    parser.add_argument(
        '--source', type=str, required=True,
        help='Original OCR folder',
    )
    parser.add_argument(
        '--cleaned', type=str, required=True, help='Cleaned output folder',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for warnings',
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    cleaned_path = Path(args.cleaned)

    if not source_path.is_dir() or not cleaned_path.is_dir():
        print('Error: Source or Cleaned paths are not valid directories.')
        return

    results = []

    # Match files in cleaned/ back to source/
    cleaned_files = list(cleaned_path.glob('*.md'))

    print(f"{'File':<30} | {'Sim':<6} | {'Delta%':<8} | {'Flags'}")
    print('-' * 80)

    for clean_file in sorted(cleaned_files):
        orig_file = source_path / clean_file.name

        if not orig_file.exists():
            continue

        orig_text = orig_file.read_text(encoding='utf-8')
        clean_text = clean_file.read_text(encoding='utf-8')

        metrics = calculate_metrics(orig_text, clean_text)
        metrics['file'] = clean_file.name
        results.append(metrics)

        flag_str = ', '.join(metrics['flags'])
        print(
            f"{clean_file.name:<30} | {metrics['similarity']:<6.4f} | {metrics['delta_percent']:>+7.2%} | {flag_str}",  # noqa: E501
        )

    # Summary Statistics
    if results:
        avg_sim = sum(r['similarity'] for r in results) / len(results)
        problematic = [r for r in results if r['flags']]

        print('\n' + '=' * 40)
        print('SUMMARY EVALUATION')
        print('=' * 40)
        print(f"Total Files Compared: {len(results)}")
        print(f"Average Similarity:   {avg_sim:.4f}")
        print(f"Files with Flags:     {len(problematic)}")

        if problematic:
            print('\nCritical Review Recommended for:')
            for r in problematic:
                print(f" - {r['file']} ({', '.join(r['flags'])})")


if __name__ == '__main__':
    main()
