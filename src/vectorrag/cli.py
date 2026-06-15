"""Command-line interface.

Usage:
    vectorrag ingest <path> [<path> ...]
    vectorrag ask "your question"
    vectorrag chat
    vectorrag info
"""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .logging_config import configure_logging
from .memory import ConversationMemory


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest.pipeline import ingest_paths

    n = ingest_paths(args.paths)
    print(f"Ingested {n} chunks.")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from .rag import RAGEngine

    engine = RAGEngine()
    resp = engine.answer(args.question)
    _print_response(resp)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from .rag import RAGEngine

    settings = get_settings()
    engine = RAGEngine(settings)
    memory = ConversationMemory(
        max_history_tokens=settings.max_history_tokens, model=settings.chat_model
    )
    print("Interactive chat. Type 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            question = input("you> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            resp = engine.answer(question, memory=memory)
            _print_response(resp)
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from .clients import build_vector_store

    settings = get_settings()
    store = build_vector_store(settings)
    print(f"Collection : {settings.collection}")
    print(f"Chroma dir : {settings.chroma_dir}")
    print(f"Chunks     : {store.count()}")
    print(f"Chat model : {settings.chat_model}")
    print(f"Embed model: {settings.embedding_model}")
    return 0


def _print_response(resp) -> None:
    print(f"\n{resp.answer}\n")
    if resp.sources:
        print("Sources:")
        for s in resp.sources:
            page = f", p.{s.page}" if s.page is not None else ""
            print(f"  [{s.id}] {s.source}{page} (score={s.score})")
    print(f"\n({resp.latency_ms} ms, grounded={resp.grounded})\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vectorrag", description="Vector RAG over 10-Q filings.")
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default WARNING).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest 10-Q files or directories.")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories (pdf/html/txt).")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a single question.")
    p_ask.add_argument("question", help="Your question.")
    p_ask.set_defaults(func=_cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive chat with memory.")
    p_chat.set_defaults(func=_cmd_chat)

    p_info = sub.add_parser("info", help="Show store/config info.")
    p_info.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
