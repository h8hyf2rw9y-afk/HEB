"""Command line: build an index, query it, evaluate it, or generate a synthetic corpus.

    python -m app.rag synthetic --out data/synthetic_policies
    python -m app.rag build --pdf-dir data/synthetic_policies --index-dir data/rag_index
    python -m app.rag query "¿Cuántos días tengo para devolver un producto?"
    python -m app.rag eval --cases tests/rag/fixtures/synthetic_eval.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import RagSettings
from .embeddings import OllamaEmbedder
from .evaluation import evaluate, load_cases
from .extraction import discover_pdfs
from .index import PolicyIndex
from .service import PolicyRetriever
from .synthetic import write_synthetic_corpus


def _settings(args: argparse.Namespace) -> RagSettings:
    s = RagSettings.from_env()
    changes = {}
    if getattr(args, "pdf_dir", None):
        changes["pdf_dir"] = Path(args.pdf_dir)
    if getattr(args, "index_dir", None):
        changes["index_dir"] = Path(args.index_dir)
    if getattr(args, "embeddings", None):
        changes["embeddings"] = args.embeddings
    return dataclasses.replace(s, **changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synthetic", help="Genera PDFs sintéticos de prueba")
    p.add_argument("--out", default="data/synthetic_policies")

    for name in ("build", "query", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("--index-dir")
        sp.add_argument("--embeddings", choices=["auto", "off", "required"])
        if name == "build":
            sp.add_argument("--pdf-dir")
        if name == "query":
            sp.add_argument("question")
            sp.add_argument("--top-k", type=int, default=None)
        if name == "eval":
            sp.add_argument("--cases", required=True)
            sp.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)

    if args.cmd == "synthetic":
        for path in write_synthetic_corpus(args.out):
            print(path)
        return 0

    settings = _settings(args)
    if args.cmd == "build":
        pdfs = discover_pdfs(settings.pdf_dir)
        if not pdfs:
            print(f"No se encontraron PDFs en {settings.pdf_dir}", file=sys.stderr)
            return 1
        embedder = None
        if settings.embeddings != "off":
            embedder = OllamaEmbedder(settings.embed_model, settings.ollama_base_url, settings.ollama_timeout_s)
        index = PolicyIndex.build(pdfs, settings, embedder)
        target = index.save(settings.index_dir)
        print(json.dumps({
            "index": str(target),
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "embed_model": index.embed_model,
            "embedding_calls": index.build_usage.calls,
            "embedding_input_tokens": index.build_usage.input_tokens,
            "warnings": index.warnings,
        }, ensure_ascii=False, indent=2))
        return 0

    retriever = PolicyRetriever.from_settings(settings)
    if args.cmd == "query":
        ans = retriever.answer(args.question, args.top_k)
        print(ans.model_dump_json(indent=2))
        return 0
    if args.cmd == "eval":
        report = evaluate(retriever, load_cases(args.cases), args.top_k)
        print(report.model_dump_json(indent=2))
        return 0 if not report.failures else 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
