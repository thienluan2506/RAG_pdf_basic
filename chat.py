"""
Set your Gemini API key in .env:
        GEMINI_API_KEY=your_key_here

    python chat.py --pdf report.pdf       # restrict to one PDF
    python chat.py --topk 8               # retrieve more chunks
    python chat.py --reindex              # re-index data/ before chatting

Commands while chatting
-----------------------
    /docs          – list indexed documents
    /pdf <name>    – switch active PDF filter  (e.g.  /pdf report.pdf)
    /pdf           – clear the filter (search all PDFs)
    /topk <n>      – change number of retrieved chunks
    /sources       – toggle showing source citations after each answer
    /clear         – clear the screen
    /help          – show this help
    /quit  or /exit or Ctrl-C  – exit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make project root importable when run directly
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from indexing import ingest, save_and_ingest
from llm import set_runtime_gemini_api_key
from rag import answer
from store import close_client, list_documents


def _print_docs(docs: list[dict]) -> None:
    if not docs:
        print(f" No documents indexed yet.")
        return
    print(f"\n Indexed documents:")
    for d in docs:
        print(
            f" {d['filename']}"
            f" ({d['chunk_count']} chunks, {d['page_count']} pages)"
        )
    print()


def _print_citations(citations: list) -> None:
    if not citations:
        return
    print(f"\n  ── Sources ──────────────────────────────────")
    for c in citations:
        line = f"  [{c.source_marker}] {c.filename}  p.{c.page}"
        if c.section:
            line += f"  │ {c.section}"
        print(f"  {line}")
    print(f"  ─────────────────────────────────────────────\n")


def _print_help() -> None:
    print(f"""
  Available commands:
  /docs          List all indexed PDFs
  /pdf <name>     Restrict answers to a specific PDF
  /pdf           Clear the PDF filter (search all)
  /topk <n>      Set number of chunks to retrieve (current: {{k}})
  /sources       Toggle source citations display
  /clear         Clear the screen
  /help          Show this help
  /quit          Exit
""")


def _clear_screen() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


# API key setup

def _resolve_api_key() -> None:
    if settings.gemini_api_key:
        set_runtime_gemini_api_key(settings.gemini_api_key)
        return

    print(f"Gemini API key not found in environment / .env file.")
    try:
        key = input("  Enter your GEMINI_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo API key provided. Exiting.")
        sys.exit(1)

    if not key:
        print(f"Empty key — cannot continue.")
        sys.exit(1)

    set_runtime_gemini_api_key(key)
    print(f"API key accepted.\n")


# Main loop

def run(
    pdf_filter: str | None = None,
    top_k: int | None = None,
    reindex: bool = False,
) -> None:
    _resolve_api_key()

    # Optional: re-index all PDFs in data/ before starting
    if reindex:
        print(f"Re-indexing PDFs in '{settings.data_dir}'")
        n = ingest(recreate=False)
        print(f"Indexed {n} chunks.\n")

    active_pdf: str | None = pdf_filter
    k: int = top_k or settings.top_k
    show_sources: bool = True

    # Show available documents on startup
    _print_docs(list_documents())

    if active_pdf:
        print(f"  Filtering to: {active_pdf}\n")

    try:
        while True:
            # Prompt
            filter_hint = f"[{active_pdf}]" if active_pdf else ""
            try:
                user_input = input(f"You{filter_hint}› ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    break

                elif cmd == "/help":
                    _print_help()

                elif cmd == "/clear":
                    _clear_screen()

                elif cmd == "/docs":
                    _print_docs(list_documents())

                elif cmd == "/pdf":
                    if arg:
                        active_pdf = arg
                        print(f"  Filter set to: {active_pdf}\n")
                    else:
                        active_pdf = None
                        print(f"  Filter clea — searching all PDFs.\n")

                elif cmd == "/topk":
                    try:
                        k = int(arg)
                        print(f"  top_k set to {k}.\n")
                    except ValueError:
                        print(f"  Usage: /topk <integer>\n")

                elif cmd == "/sources":
                    show_sources = not show_sources
                    state = "ON" if show_sources else "OFF"
                    print(f"  Source display: {state}\n")

                else:
                    print(f"  Unknown command '{cmd}'. Type /help for help.\n")

                continue

            # RAG answer
            filters = {"filename": active_pdf} if active_pdf else None

            print(f"\nSearching {k} chunks…")
            try:
                result = answer(question=user_input, k=k, filters=filters)
            except RuntimeError as exc:
                print(f"\nError: {exc}\n")
                continue

            # Print answer
            print(f"\nAssistant› {result.answer}\n")

            # Print citations
            if show_sources and result.citations:
                _print_citations(result.citations)

    except KeyboardInterrupt:
        pass

    finally:
        close_client()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive RAG CLI — ask questions about indexed PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        metavar="FILENAME",
        default=None,
        help="Restrict answers to a specific PDF (e.g. --pdf report.pdf)",
    )
    parser.add_argument(
        "--topk",
        metavar="N",
        type=int,
        default=None,
        help=f"Number of chunks to retrieve (default: {settings.top_k})",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index all PDFs in data/ before starting the chat",
    )
    parser.add_argument(
        "--ingest-pdf",
        metavar="PATH",
        default=None,
        help="Ingest a single PDF file then start chatting (e.g. --ingest-pdf ./my.pdf)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    # Optional: ingest a single PDF passed on the command line
    if args.ingest_pdf:
        pdf_path = Path(args.ingest_pdf)
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            sys.exit(1)
        print(f"Ingesting '{pdf_path.name}' …")
        result = save_and_ingest(pdf_path.read_bytes(), pdf_path.name)
        print(f"Indexed {result['chunks_indexed']} chunks from '{result['filename']}'.\n")

    run(
        pdf_filter=args.pdf,
        top_k=args.topk,
        reindex=args.reindex,
    )