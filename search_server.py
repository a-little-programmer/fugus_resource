#!/usr/bin/env python3
"""Small standard-library web server for the taxon retrieval demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.alias_matcher import load_alias_dict
from src.embedder import DEFAULT_FINE_TUNED_MODEL, Embedder, create_embedder
from src.entity_index import EntityIndex
from src.search_service import retrieve


DEFAULT_ALIAS_PATH = "artifacts/alias_dict.json"
DEFAULT_INDEX_PATH = "artifacts/species_index.pkl"
DEFAULT_MODEL_NAME = DEFAULT_FINE_TUNED_MODEL
DEFAULT_THRESHOLD = 0.82
WEB_ROOT = Path(__file__).resolve().parent / "web"


class SearchApp:
    def __init__(
        self,
        alias_path: str,
        index_path: str,
        model_name: str,
        backend: str,
        threshold: float,
    ) -> None:
        self.alias_path = alias_path
        self.index_path = index_path
        self.model_name = model_name
        self.backend = backend
        self.threshold = threshold
        self.alias_dict = load_alias_dict(alias_path)
        self.index = EntityIndex.load(index_path)
        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.index.model_name or self.model_name, self.backend)
        return self._embedder

    def search(self, query: str, top_k: int, threshold: float | None = None) -> dict:
        return retrieve(
            query=query,
            alias_dict=self.alias_dict,
            index=self.index,
            embedder=lambda: self.embedder,
            top_k=top_k,
            threshold=self.threshold if threshold is None else threshold,
        )

    def status(self) -> dict:
        return {
            "entities": len(self.index.entities),
            "name_vectors": len(self.index.records),
            "index_backend": self.index.backend_name,
            "model_name": self.index.model_name,
            "server_backend": self.backend,
            "threshold": self.threshold,
        }


def make_handler(app: SearchApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/search":
                self.handle_search(parsed.query)
                return
            if parsed.path == "/api/status":
                self.write_json(app.status())
                return
            self.serve_static(parsed.path)

        def handle_search(self, query_string: str) -> None:
            params = parse_qs(query_string)
            query = (params.get("q") or [""])[0].strip()
            if not query:
                self.write_json({"message": "Query is required.", "results": []}, status=400)
                return

            top_k = parse_int((params.get("top_k") or ["5"])[0], default=5)
            threshold = parse_float((params.get("threshold") or [""])[0], default=app.threshold)
            try:
                payload = app.search(query, top_k=top_k, threshold=threshold)
            except Exception as exc:
                self.write_json({"message": str(exc), "results": []}, status=500)
                return
            self.write_json(payload)

        def serve_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (WEB_ROOT / relative).resolve()
            if WEB_ROOT not in target.parents and target != WEB_ROOT:
                self.send_error(403)
                return
            if not target.exists() or not target.is_file():
                self.send_error(404)
                return

            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def write_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def parse_int(value: str, default: int) -> int:
    try:
        return max(1, int(value))
    except ValueError:
        return default


def parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except ValueError:
        return default


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the taxon retrieval web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--alias", default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sentence-transformers", "char-ngram"),
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    app = SearchApp(args.alias, args.index, args.model, args.backend, args.threshold)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving taxon retrieval UI at http://{args.host}:{args.port}")
    print(f"Loaded {len(app.index.entities)} entities and {len(app.index.records)} name vectors")
    server.serve_forever()


if __name__ == "__main__":
    main()
