from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Literal

import httpx
from openai import AsyncOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel

# Configure logger
logger = logging.getLogger(__name__)


SUMMARIZE_PROMPT_TEMPLATE = """You are an expert summarizer. Your task is to summarize the provided text.

Follow these strict rules:
1. EXACT LENGTH: Your summary MUST be exactly two sentences long. No more, no less.
2. DETECT CONTINUATION: Look for clues that this text is a continuation of a larger document, such as starting mid-sentence OR beginning with isolated metadata (e.g., standalone page numbers, dates, or document titles). If you detect these clues, IGNORE the metadata and begin your first sentence with the exact phrase: "Continuing from the previous page, ..."
3. PLAIN TEXT ONLY: Do NOT use any markdown formatting in your output (no **bold**, *italics*, or lists).
4. OUTPUT FORMAT: Output ONLY the two-sentence summary. Do NOT include any intro or outro text (e.g., "Here is the summary:").

Text to summarize:
<text>
{content}
</text>
"""  # noqa: E501

STRUCTURE_PROMPT_TEMPLATE = """You are an expert archivist sorting a continuous deck of documents titled "{deck_title}".
Your task is to analyze the current page summary and decide if it continues the previous document or starts a new one.

Follow these strict rules:
1. DECISION LOGIC:
   - IF the current page is a direct continuation of the previous page (e.g., page 2 of a letter, report, or continuous narrative) -> set "action" to "APPEND".
   - IF the current page shifts to a distinct new letter, report, or separate item -> set "action" to "NEW_DOCUMENT".
2. TITLE FORMAT: Titles MUST be formatted exactly like this: [Year/Month] - [Author/Sender to Recipient] - [Short Summary].
   - EXAMPLES:
     * [Undated] - Law Library Administration: Course Project
     * 1990 - Goldsmith - Excerpts from History of Law Librarianship
     * 1989/10 - Hazelton to Chrisholm - attached with the LIS 577 project
3. SUMMARY LENGTH: The "Short Summary" portion of the title MUST be 5 to 7 words maximum. No exceptions.
4. TITLE UPDATES: If appending, use the previous document's title (only tweak it if critically necessary). If it is a new document, generate a completely new title based on the format.
5. OUTPUT FORMAT: Output ONLY valid JSON. Do NOT wrap the JSON in ```json backticks. Do NOT include any intro or conversational filler.

Context Data:
<recent_documents>
{recent_docs_json}
</recent_documents>

<previous_page_summary>
{last_summary_text}
</previous_page_summary>

<current_page_summary>
{page_num_str}: {summary}
</current_page_summary>
"""  # noqa: E501

REPAIR_PROMPT_TEMPLATE = """You are an expert copyeditor. Your task is to repair OCR errors in the provided markdown text, which is part of a document titled "{doc_title}".

Follow these strict rules:
1. FIX TYPOS: Correct misspellings, character confusion (rn/m, 1/l, 0/O), missing/extra spaces, and split hyphenated words.
2. REFLOW TEXT: Remove unwanted line breaks within paragraphs.
3. CLEAN ARTIFACTS: Delete stray symbols (~, |, ^) and duplicated text blocks caused by OCR errors.
4. FIX MARKDOWN: Restore broken markdown syntax (lists, bold, italics). Convert raw HTML `<img>` tags to markdown `![image]()` or remove them if they are garbage.
5. FIX AND FILTER HEADINGS: All valid headings must be Level 3 (###) or lower. DELETE false or nonsensical headings (e.g., standalone page numbers, random characters, or stray fragments accidentally formatted as headers by the OCR).
6. BE FAITHFUL: Do NOT add new information, hallucinate, or rewrite sentences. Only fix formatting and OCR errors.
7. OUTPUT FORMAT: Output ONLY the repaired text. Do NOT wrap the output in ```markdown ...
``` backticks. Do NOT include any intro or outro text (e.g., "Here is the repaired text:").

Text to repair:
<text>
{content}
</text>
"""  # noqa: E501


class StructureDecision(BaseModel):
    action: Literal['APPEND', 'NEW_DOCUMENT']
    document_title: str


def setup_logging(debug: bool) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(message)s',
        stream=sys.stderr,
    )
    # Silence third-party loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)


async def check_health(base_url: str) -> bool:
    """Verify the local LLM server's health."""
    health_url = base_url.replace(
        '/v1', '/health',
    ) if '/v1' in base_url else f"{base_url}/health"
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(health_url, timeout=2.0)
            if response.status_code == 200:
                logger.info('✅ Server Health: OK')
                return True
    except Exception:
        try:
            temp_client = AsyncOpenAI(
                base_url=base_url, api_key='local-server',
            )
            await temp_client.models.list()
            logger.info('✅ Server Health: OK (via /v1/models)')
            return True
        except Exception as e:
            logger.error(f"❌ Server Health Check failed at {base_url}: {e}")
            return False
    return False


async def summarize_file(
    client: AsyncOpenAI,
    file_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    """Summarize a single file using the LLM with concurrency control."""
    async with semaphore:
        try:
            content = file_path.read_text(encoding='utf-8')
            if not content.strip():
                return {
                    'file_name': file_path.name,
                    'summary': 'Skipping: Empty file.',
                }

            is_debug = logger.isEnabledFor(logging.DEBUG)
            if is_debug:
                logger.debug(f"\n--- STREAMING START: {file_path.name} ---")

            full_summary = ''

            stream = await client.chat.completions.create(
                model='local-model',
                messages=[
                    {
                        'role': 'user',
                        'content': SUMMARIZE_PROMPT_TEMPLATE.format(
                            content=content,
                        ),
                    },
                ],
                max_tokens=300,
                temperature=0.1,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Capture standard content
                if delta.content:
                    full_summary += delta.content
                    if is_debug:
                        sys.stdout.write(delta.content)
                        sys.stdout.flush()

            if is_debug:
                logger.debug(f"\n--- STREAMING END: {file_path.name} ---\n")

            if not full_summary.strip():
                return {
                    'file_name': file_path.name,
                    'summary': '[Warning: Model returned empty content.]',
                }

            return {
                'file_name': file_path.name,
                'summary': full_summary.strip(),
            }
        except Exception as e:
            return {
                'file_name': file_path.name,
                'summary': f"Error processing file: {e}",
            }


async def generate_structure_sequentially(
    client: AsyncOpenAI,
    sorted_summaries: list[dict[str, str]],
    model: str,
    deck_title: str,
    root_path: Path,
) -> None:
    """
    Generate the overall document structure sequentially based on summaries.
    """
    print('\n' + '='*40)
    print('GENERATING OVERALL STRUCTURE')
    print('='*40)

    structure_file = root_path / 'structure.json'

    # Try to load existing structure
    state: dict[str, str | list[dict[str, str | list[str]]]] = {
        'deck_title': deck_title,
        'documents': [],
    }

    if structure_file.exists():
        try:
            state = json.loads(structure_file.read_text(encoding='utf-8'))
            print(f"Loaded existing structure from {structure_file.name}")
        except Exception as e:
            print(f"Failed to load structure checkpoint: {e}")

    # Track processed pages to allow resuming
    processed_pages = set()
    docs = state.get('documents', [])
    assert isinstance(docs, list)
    for doc in docs:
        pages = doc.get('pages', [])
        assert isinstance(pages, list)
        processed_pages.update(pages)

    last_page_summary: str | None = None

    for page in sorted_summaries:
        file_name = page.get('file_name', '')
        summary = page.get('summary', '')
        if not file_name or 'Skipping' in summary or 'Error' in summary:
            continue

        if file_name in processed_pages:
            last_page_summary = summary
            continue

        docs_for_context = state.get('documents', [])
        assert isinstance(docs_for_context, list)
        recent_docs = docs_for_context[-5:] if docs_for_context else []

        page_match = re.search(r'_(\d+)\.md$', file_name)
        page_num_str = f"(Page {int(page_match.group(1))})" if page_match else ''  # noqa: E501

        prompt = STRUCTURE_PROMPT_TEMPLATE.format(
            deck_title=deck_title,
            recent_docs_json=json.dumps(recent_docs),
            last_summary_text=last_page_summary,
            page_num_str=page_num_str,
            summary=summary,
        )

        try:
            response = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                max_tokens=800,
                temperature=0.1,
                response_format=StructureDecision,
            )

            result = response.choices[0].message.parsed
            if result:
                action = result.action
                doc_title = result.document_title

                docs = state['documents']
                assert isinstance(docs, list)

                if action == 'NEW_DOCUMENT' or not docs:
                    docs.append({
                        'title': doc_title,
                        'pages': [file_name],
                    })
                    print(f"{file_name} -> NEW_DOCUMENT: \"{doc_title}\"")
                else:
                    # APPEND
                    docs[-1]['title'] = doc_title
                    pages = docs[-1]['pages']
                    assert isinstance(pages, list)
                    pages.append(file_name)
                    print(f"{file_name} -> APPEND to: \"{doc_title}\"")

                # Save checkpoint after each successful page
                try:
                    structure_file.write_text(
                        json.dumps(
                            state, indent=2,
                        ), encoding='utf-8',
                    )
                except Exception as e:
                    print(f"Failed to save structure checkpoint: {e}")
        except LengthFinishReasonError as e:
            print(f"Error processing {file_name} for structure: {e}")
            try:
                partial_content = e.completion.choices[0].message.content
                print(f"\n--- DEBUG: TRUNCATED OUTPUT FOR {file_name} ---")
                print(partial_content)
                print('--- END DEBUG ---\n')
            except Exception:
                pass
        except Exception as e:
            print(f"Error processing {file_name} for structure: {e}")

        last_page_summary = summary

    # Final save just to be sure
    try:
        structure_file.write_text(
            json.dumps(
                state, indent=2,
            ), encoding='utf-8',
        )
    except Exception as e:
        print(f"Failed to save final structure: {e}")

    # Output final structure
    print('\n' + '='*40)
    print('FINAL DOCUMENT STRUCTURE')
    print('='*40)
    print(f"# {state.get('deck_title', 'Untitled File Deck')}\n")
    docs = state.get('documents', [])
    assert isinstance(docs, list)
    for doc in docs:
        pages = doc.get('pages', [])
        assert isinstance(pages, list)
        if not pages:
            continue
        page_range = f"{pages[0]} to {pages[-1]}" if len(
            pages,
        ) > 1 else pages[0]
        print(f"## {doc.get('title', 'Untitled Document')} "
              f"(Pages: {page_range})")


async def repair_file(
    client: AsyncOpenAI,
    file_name: str,
    doc_title: str,
    root_path: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """Repair a single file and save it to the cleaned directory."""
    cleaned_dir = root_path / 'cleaned'
    out_file = cleaned_dir / file_name
    if out_file.exists():
        return

    file_path = root_path / file_name
    async with semaphore:
        try:
            content = file_path.read_text(encoding='utf-8')
            if not content.strip():
                return

            print(f"Repairing {file_name}...")

            prompt = REPAIR_PROMPT_TEMPLATE.format(
                doc_title=doc_title,
                content=content,
            )

            response = await client.chat.completions.create(
                model='local-model',
                messages=[
                    {'role': 'user', 'content': prompt},
                ],
                max_tokens=4096,
                temperature=0.1,
            )

            choice = response.choices[0]
            if choice.finish_reason == 'length':
                raise ValueError(
                    'Output truncated: max_tokens limit reached.',
                )

            repaired_content = choice.message.content or ''
            out_file.write_text(repaired_content.strip(), encoding='utf-8')
            print(f"✅ Repaired {file_name}")
        except Exception as e:
            print(f"❌ Error repairing {file_name}: {e}")


async def repair_document_pages(
    client: AsyncOpenAI,
    root_path: Path,
    concurrency: int,
) -> None:
    """Read the structure and repair all pages."""
    print('\n' + '='*40)
    print('REPAIRING DOCUMENT PAGES')
    print('='*40)

    structure_file = root_path / 'structure.json'
    if not structure_file.exists():
        print('No structure.json found, skipping repair.')
        return

    try:
        state = json.loads(structure_file.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load structure checkpoint: {e}")
        return

    cleaned_dir = root_path / 'cleaned'
    cleaned_dir.mkdir(exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    docs = state.get('documents', [])
    for doc in docs:
        doc_title = doc.get('title', 'Untitled Document')
        pages = doc.get('pages', [])
        for file_name in pages:
            tasks.append(
                repair_file(
                    client, file_name,
                    doc_title, root_path, semaphore,
                ),
            )

    if not tasks:
        print('No pages found to repair.')
        return

    print(f"Starting repair of {len(tasks)} pages...")
    await asyncio.gather(*tasks)
    print('Done repairing pages.')


def combine_cleaned_pages(root_path: Path) -> None:
    """Join all cleaned pages into a single combined markdown file."""
    structure_file = root_path / 'structure.json'
    if not structure_file.exists():
        print('No structure.json found, skipping combination.')
        return

    try:
        state = json.loads(structure_file.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Failed to load structure for combination: {e}")
        return

    deck_title = state.get('deck_title', 'Untitled File Deck')
    docs = state.get('documents', [])
    if not docs:
        print('No documents found in structure, skipping combination.')
        return

    # Determine common prefix from the first file name
    prefix = ''
    first_page = docs[0].get('pages', [None])[0]
    if first_page:
        match = re.match(r'^(.*)_\d+\.md$', first_page)
        if match:
            prefix = match.group(1)

    output_filename = f"{prefix}_cleaned_combined.md" if prefix else '_cleaned_combined.md'  # noqa: E501
    output_path = root_path / output_filename

    print(f"\nCombining pages into {output_path.name}...")

    combined_content = [f"# {deck_title}\n"]
    cleaned_dir = root_path / 'cleaned'

    for doc in docs:
        doc_title = doc.get('title', 'Untitled Document')
        pages = doc.get('pages', [])
        if not pages:
            continue

        combined_content.append(f"## {doc_title}\n")

        for page_name in pages:
            page_file = cleaned_dir / page_name
            if page_file.exists():
                combined_content.append(f"<!-- Page: {page_name} -->")
                content = page_file.read_text(encoding='utf-8').strip()
                combined_content.append(content)
                combined_content.append('')  # Extra newline between pages

    try:
        output_path.write_text('\n'.join(combined_content), encoding='utf-8')
        print(f"✅ Successfully created {output_path.name}")
    except Exception as e:
        print(f"❌ Failed to write combined file: {e}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='OCR Cleanup and Summarization using local LLM',
    )
    parser.add_argument(
        '--path', type=str,
        required=True, help='Path to markdown files',
    )
    parser.add_argument(
        '--deck-title', type=str,
        required=True, help='The overall title of the document deck',
    )
    parser.add_argument(
        '--base-url', type=str,
        default='http://localhost:8080/v1', help='API Base URL',
    )
    parser.add_argument(
        '--concurrency', type=int,
        default=2, help='Max concurrent requests',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging and streaming',
    )
    args = parser.parse_args()

    setup_logging(args.debug)
    client = AsyncOpenAI(base_url=args.base_url, api_key='local-server')

    if not await check_health(args.base_url):
        return

    root_path = Path(args.path)
    if not root_path.is_dir():
        logger.error(f"Error: {args.path} is not a directory.")
        return

    pattern = re.compile(r'.*_\d{2}\.md$')
    files_to_process = [
        f for f in root_path.glob(
            '*.md',
        ) if pattern.match(f.name)
    ]

    if not files_to_process:
        logger.warning(
            f"No files matching pattern *_XX.md found in {args.path}",
        )
        return

    checkpoint_file = root_path / 'summaries_checkpoint.json'
    cached_summaries = {}
    if checkpoint_file.exists():
        try:
            cached_summaries = json.loads(
                checkpoint_file.read_text(encoding='utf-8'),
            )
            logger.info(
                f"Loaded {len(cached_summaries)} cached summaries "
                'from {checkpoint_file.name}',
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    pending_files = [
        f for f in files_to_process if f.name not in cached_summaries
    ]

    logger.info(
        f"Found {len(files_to_process)} total files. "
        f"{len(pending_files)} pending processing "
        f"(concurrency={args.concurrency})...",
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [summarize_file(client, f, semaphore) for f in pending_files]

    new_results = await asyncio.gather(*tasks)

    results = [
        {'file_name': k, 'summary': v}
        for k, v in cached_summaries.items()
    ]
    results.extend(new_results)

    updated_cache = {r['file_name']: r['summary'] for r in results}
    try:
        checkpoint_file.write_text(
            json.dumps(
                updated_cache, indent=2,
            ), encoding='utf-8',
        )
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

    print('\n' + '='*40)
    print('FINAL SUMMARIES')
    print('='*40)
    for result in results:
        print(f"--- {result['file_name']} ---")
        print(f"{result['summary']}\n")

    # Sort summaries naturally by file number
    def extract_number(f: dict[str, str]) -> int:
        match = re.search(r'_(\d+)\.md$', f.get('file_name', ''))
        return int(match.group(1)) if match else 0

    sorted_summaries = sorted(results, key=extract_number)

    await generate_structure_sequentially(
        client,
        sorted_summaries, 'local-model',
        args.deck_title, root_path,
    )

    await repair_document_pages(client, root_path, args.concurrency)

    combine_cleaned_pages(root_path)


if __name__ == '__main__':
    asyncio.run(main())
