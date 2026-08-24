#!/usr/bin/env python3
"""Microsoft GraphRAG 3.1.1 adapter for the RAG evaluation v2 contract."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai_embeddings import (  # noqa: E402
    OpenAIEmbeddingClient,
    codex_cli_environment,
    codex_evaluator_isolation_args,
)

import pandas as pd
import yaml
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, load_search_prompt


PINNED_VERSION = "3.1.1"
FRAMEWORK_ID = "microsoft-graphrag"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
COMPLETION_EXECUTION_MODES = ("codex-exec", "anthropic-api")
COMPLETION_MODELS = {
    "codex-exec": "gpt-5.6-terra",
    "anthropic-api": "claude-sonnet-5",
}
ANTHROPIC_TIMEOUT_SECONDS = 600.0


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_command = subparsers.add_parser("doctor")
    doctor_command.add_argument(
        "--completion-execution",
        choices=COMPLETION_EXECUTION_MODES,
        default="codex-exec",
    )
    for name in ("build", "retrieve"):
        command = subparsers.add_parser(name)
        add_common_arguments(command)
        if name == "retrieve":
            command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        doctor(args.completion_execution)
    elif args.command == "build":
        build(args)
    else:
        retrieve(args)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-dimensions", required=True, type=int)
    parser.add_argument("--completion-model", required=True)
    parser.add_argument("--completion-reasoning-effort", required=True)
    parser.add_argument(
        "--completion-execution", required=True, choices=COMPLETION_EXECUTION_MODES
    )
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--query-concurrency", required=True, type=int)
    parser.add_argument("--index-source-dir")


def doctor(completion_execution: str = "codex-exec") -> None:
    installed = version("graphrag")
    issues: list[str] = []
    if installed != PINNED_VERSION:
        issues.append(f"expected graphrag {PINNED_VERSION}, found {installed}")
    issues.extend(completion_execution_issues(completion_execution))
    if shutil.which("graphrag") is None:
        issues.append("graphrag CLI is not on PATH")
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY is not set")
    print(json.dumps({
        "status": "blocked" if issues else "ready",
        "version": installed,
        "detail": "; ".join(issues) if issues else "official standard index and native local-search context builder",
    }))


def completion_execution_issues(completion_execution: str) -> list[str]:
    issues: list[str] = []
    if completion_execution == "anthropic-api":
        if not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("ANTHROPIC_API_KEY is not set")
        if not anthropic_package_available():
            issues.append("anthropic package is not importable")
    else:
        if shutil.which("codex") is None:
            issues.append("codex CLI is not on PATH")
    return issues


def anthropic_package_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def validate_shared_options(args: argparse.Namespace) -> None:
    if args.completion_execution not in COMPLETION_MODELS:
        raise ValueError(
            f"completion_execution must be one of {sorted(COMPLETION_MODELS)}, "
            f"found {args.completion_execution!r}"
        )
    expected = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "completion_model": COMPLETION_MODELS[args.completion_execution],
        "completion_reasoning_effort": "medium",
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"{name} must be {value!r}, found {getattr(args, name)!r}")


def build(args: argparse.Namespace) -> None:
    validate_shared_options(args)
    if args.index_source_dir:
        raise ValueError("build does not accept --index-source-dir")
    dataset_dir = Path(args.dataset_dir)
    index_dir = Path(args.index_dir)
    corpus = read_jsonl(dataset_dir / "corpus.jsonl")
    digest = record_digest(corpus)
    marker_path = index_dir / "rag-eval-build.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if (
            marker.get("corpusDigest") == digest
            and marker.get("embeddingModel") == EMBEDDING_MODEL
            and marker.get("embeddingDimensions") == EMBEDDING_DIMENSIONS
        ):
            return
    index_dir.mkdir(parents=True, exist_ok=True)
    initialize_project(index_dir, args)
    write_input_documents(index_dir / "input", corpus)
    with openai_compatible_server(args) as api_base:
        configure_project(index_dir, args, api_base)
        result = subprocess.run(
            ["graphrag", "index", "--root", str(index_dir)],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"graphrag index failed:\n{result.stderr}\n{result.stdout}")
    write_json_atomic(marker_path, {
        "framework": FRAMEWORK_ID,
        "version": PINNED_VERSION,
        "corpusDigest": digest,
        "documents": len(corpus),
        "embeddingMode": "framework-native symmetric text embedding",
        "embeddingModel": EMBEDDING_MODEL,
        "embeddingDimensions": EMBEDDING_DIMENSIONS,
        "completionExecution": args.completion_execution,
        "completionModel": args.completion_model,
    })


def retrieve(args: argparse.Namespace) -> None:
    validate_shared_options(args)
    dataset_dir = Path(args.dataset_dir)
    index_dir = Path(args.index_dir)
    if args.query_concurrency <= 0:
        raise ValueError("query_concurrency must be positive")
    if args.index_source_dir:
        index_source_dir = Path(args.index_source_dir).resolve()
        validate_warm_index(index_source_dir, read_jsonl(dataset_dir / "corpus.jsonl"))
        prepare_warm_runtime(index_source_dir, index_dir)
    queries = read_jsonl(dataset_dir / "queries.jsonl")
    with openai_compatible_server(args) as api_base:
        configure_project(index_dir, args, api_base)
        config = load_config(index_dir)
        engine = create_context_engine(config, index_dir)
        records: list[dict[str, Any]] = []
        for query in queries:
            started = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                context = engine.context_builder.build_context(
                    query=query["text"],
                    **{
                        **engine.context_builder_params,
                        "top_k_mapped_entities": args.top_k,
                        "top_k_relationships": args.top_k,
                    },
                )
                text = context.context_chunks
                evidence = [] if not text.strip() else [{
                    "id": f"{FRAMEWORK_ID}:{query['id']}:context",
                    "sourceId": "graphrag-local-search-context",
                    "text": text,
                    "score": 1,
                    "rank": 1,
                    "metadata": {
                        "searchMethod": "local",
                        "nativeContext": True,
                        "contextTables": ",".join(sorted(context.context_records.keys())),
                    },
                }]
                records.append(
                    retrieval_record(args, query["id"], "ok", evidence, started, started_at, None)
                )
            except Exception as error:  # preserve per-query failures
                records.append(
                    retrieval_record(
                        args, query["id"], "error", [], started, started_at, str(error)
                    )
                )
    write_jsonl(Path(args.output), records)


def initialize_project(index_dir: Path, args: argparse.Namespace) -> None:
    if (index_dir / "settings.yaml").exists():
        return
    result = subprocess.run([
        "graphrag", "init", "--root", str(index_dir),
        "--model", args.completion_model,
        "--embedding", args.embedding_model,
    ], text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"graphrag init failed:\n{result.stderr}\n{result.stdout}")


def configure_project(index_dir: Path, args: argparse.Namespace, api_base: str) -> None:
    settings_path = index_dir / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    completion = settings["completion_models"]["default_completion_model"]
    completion.update({
        "model_provider": "openai",
        "model": args.completion_model,
        "api_key": f"local-{args.completion_execution}",
        "api_base": f"{api_base}/v1",
    })
    embedding = settings["embedding_models"]["default_embedding_model"]
    embedding.update({
        "model_provider": "openai",
        "model": args.embedding_model,
        "api_key": "local-openai-embedding-proxy",
        "api_base": f"{api_base}/v1",
        "call_args": {"dimensions": EMBEDDING_DIMENSIONS},
    })
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))


def write_input_documents(directory: Path, corpus: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, document in enumerate(corpus):
        name = f"{index:06d}-{hashlib.sha256(document['id'].encode()).hexdigest()[:16]}.txt"
        (directory / name).write_text(
            f"title: {document['title']}\nsource_id: {document['sourceId']}\n\n{document['text']}"
        )


def create_context_engine(config: Any, index_dir: Path) -> Any:
    output = index_dir / "output"
    entities = pd.read_parquet(output / "entities.parquet")
    communities = pd.read_parquet(output / "communities.parquet")
    reports = pd.read_parquet(output / "community_reports.parquet")
    text_units = pd.read_parquet(output / "text_units.parquet")
    relationships = pd.read_parquet(output / "relationships.parquet")
    covariate_path = output / "covariates.parquet"
    covariates = pd.read_parquet(covariate_path) if covariate_path.exists() else None
    community_level = 2
    store = get_embedding_store(
        config=config.vector_store,
        embedding_name=entity_description_embedding,
    )
    entities_ = read_indexer_entities(entities, communities, community_level)
    covariates_ = read_indexer_covariates(covariates) if covariates is not None else []
    prompt = load_search_prompt(config.local_search.prompt)
    return get_local_search_engine(
        config=config,
        reports=read_indexer_reports(reports, communities, community_level),
        text_units=read_indexer_text_units(text_units),
        entities=entities_,
        relationships=read_indexer_relationships(relationships),
        covariates={"claims": covariates_},
        description_embedding_store=store,
        response_type="Multiple Paragraphs",
        system_prompt=prompt,
        callbacks=None,
    )


@contextmanager
def openai_compatible_server(args: argparse.Namespace) -> Iterator[str]:
    backend = LocalModelBackend(args)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(length))
                if self.path.endswith("/chat/completions"):
                    response = backend.chat_completion(request)
                elif self.path.endswith("/embeddings"):
                    response = backend.embeddings(request)
                else:
                    self.send_error(404)
                    return
                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:
                payload = json.dumps({"error": {"message": str(error), "type": "adapter_error"}}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class LocalModelBackend:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.completion_slots = threading.Semaphore(4)
        self.embedding_client = OpenAIEmbeddingClient(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            usage_path=Path(args.index_dir) / "openai-embedding-usage.jsonl",
        )

    def chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("stream"):
            raise ValueError("streaming chat completions are not supported during indexing")
        messages = request.get("messages") or []
        response_format = request.get("response_format") or {}
        complete = (
            anthropic_complete_sync
            if self.args.completion_execution == "anthropic-api"
            else codex_complete_sync
        )
        with self.completion_slots:
            content = complete(
                messages,
                self.args.completion_model,
                self.args.completion_reasoning_effort,
                response_format,
            )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.args.completion_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        dimensions = int(request.get("dimensions") or EMBEDDING_DIMENSIONS)
        if dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding dimensions must be {EMBEDDING_DIMENSIONS}")
        inputs = request.get("input")
        texts = [inputs] if isinstance(inputs, str) else inputs
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding input must be a string or string list")
        vectors, usage = self.embedding_client.embed_with_usage(texts)
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors)],
            "model": EMBEDDING_MODEL,
            "usage": usage.as_openai(),
        }


def codex_complete_sync(
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    response_format: dict[str, Any],
) -> str:
    rendered = "\n".join(
        f"<{message.get('role', 'user')}>\n{render_message_content(message.get('content'))}\n</{message.get('role', 'user')}>"
        for message in messages
    )
    json_schema = response_format.get("json_schema", {}).get("schema")
    structured_instruction = ""
    if json_schema:
        structured_instruction = f"\nReturn valid JSON matching this schema exactly:\n{json.dumps(json_schema)}"
    elif response_format.get("type") == "json_object":
        structured_instruction = "\nReturn one valid JSON object without a markdown fence."
    prompt = "\n".join([
        "Act as a deterministic completion backend for the embedded messages.",
        "Do not use tools, files, web search, or prior conversation.",
        "Put the exact completion in the JSON field `text`.",
        f"<messages>\n{rendered}\n</messages>",
        structured_instruction,
    ])
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
    }
    with tempfile.TemporaryDirectory(prefix="rag-eval-graphrag-") as directory:
        schema_path = Path(directory) / "schema.json"
        output_path = Path(directory) / "output.json"
        schema_path.write_text(json.dumps(schema))
        result = subprocess.run([
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            *codex_evaluator_isolation_args(),
            "--skip-git-repo-check", "--sandbox", "read-only", "--json",
            "--model", model, "--config", f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema", str(schema_path), "--output-last-message", str(output_path), "-",
        ], input=prompt, text=True, capture_output=True, timeout=600,
            env=codex_cli_environment())
        if result.returncode != 0:
            raise RuntimeError(f"codex exec failed: {result.stderr}\n{result.stdout}")
        text = json.loads(output_path.read_text())["text"]
        if json_schema or response_format.get("type") == "json_object":
            json.loads(text)
        return text


_anthropic_client: Any | None = None
_anthropic_client_lock = threading.Lock()


def anthropic_client() -> Any:
    """Lazily build one in-process Anthropic client (reads ANTHROPIC_API_KEY).

    Unlike the codex path, no environment scrubbing applies here: the call runs
    in-process and must see ANTHROPIC_API_KEY in os.environ.
    """
    global _anthropic_client
    with _anthropic_client_lock:
        if _anthropic_client is None:
            from anthropic import Anthropic

            _anthropic_client = Anthropic(
                max_retries=0, timeout=ANTHROPIC_TIMEOUT_SECONDS
            )
        return _anthropic_client


def anthropic_request_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Map OpenAI-style chat messages onto (system prompt, Anthropic messages)."""
    system_parts: list[str] = []
    chat: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = render_message_content(message.get("content"))
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            chat.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported chat message role: {role!r}")
    return "\n".join(system_parts), chat


def anthropic_complete_sync(
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    response_format: dict[str, Any],
) -> str:
    json_schema = response_format.get("json_schema", {}).get("schema")
    structured_instruction = ""
    if json_schema:
        structured_instruction = f"Return valid JSON matching this schema exactly:\n{json.dumps(json_schema)}"
    elif response_format.get("type") == "json_object":
        structured_instruction = "Return one valid JSON object without a markdown fence."
    system, chat = anthropic_request_messages(messages)
    if structured_instruction:
        system = f"{system}\n{structured_instruction}".strip()
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": 16000,
        "messages": chat,
        "output_config": {"effort": reasoning_effort},
    }
    if system:
        request["system"] = system
    response = anthropic_client().messages.create(**request)
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(
            f"anthropic completion stopped early: stop_reason={response.stop_reason!r}"
        )
    text = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    if json_schema or response_format.get("type") == "json_object":
        json.loads(text)
    return text


def render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def retrieval_record(
    args: argparse.Namespace,
    query_id: str,
    status: str,
    evidence: list[dict[str, Any]],
    started: float,
    started_at: str,
    error: str | None,
) -> dict[str, Any]:
    dataset_id = Path(args.dataset_dir).parents[1].name
    return {
        "datasetId": dataset_id,
        "frameworkId": FRAMEWORK_ID,
        "queryId": query_id,
        "status": status,
        "evidence": evidence,
        "latencyMs": (time.perf_counter() - started) * 1000,
        "inputTokens": None,
        "error": error,
        "frameworkVersion": PINNED_VERSION,
        "configDigest": "set-by-parent-harness",
        "startedAt": started_at,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }


def validate_warm_index(index_source_dir: Path, corpus: list[dict[str, Any]]) -> None:
    marker_path = index_source_dir / "rag-eval-build.json"
    if not marker_path.is_file():
        raise RuntimeError(f"warm index marker missing: {marker_path}")
    marker = json.loads(marker_path.read_text())
    expected = {
        "framework": FRAMEWORK_ID,
        "version": PINNED_VERSION,
        "corpusDigest": record_digest(corpus),
        "embeddingModel": EMBEDDING_MODEL,
        "embeddingDimensions": EMBEDDING_DIMENSIONS,
    }
    mismatches = [
        f"{name}: expected {value!r}, found {marker.get(name)!r}"
        for name, value in expected.items()
        if marker.get(name) != value
    ]
    if mismatches:
        raise RuntimeError("warm index marker mismatch: " + "; ".join(mismatches))


def prepare_warm_runtime(index_source_dir: Path, index_dir: Path) -> None:
    """Expose a completed GraphRAG output through a clean runtime directory."""
    if index_source_dir == index_dir.resolve():
        raise RuntimeError("warm index source and clean runtime must be different directories")
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in ("output", "input", "prompts", "rag-eval-build.json"):
        source = index_source_dir / name
        if not source.exists():
            if name in {"input", "prompts"}:
                continue
            raise RuntimeError(f"warm index component missing: {source}")
        target = index_dir / name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise RuntimeError(f"clean runtime collision: {target}")
            continue
        target.symlink_to(source, target_is_directory=source.is_dir())
    settings_source = index_source_dir / "settings.yaml"
    settings_target = index_dir / "settings.yaml"
    if not settings_source.is_file():
        raise RuntimeError(f"warm index settings missing: {settings_source}")
    if not settings_target.exists():
        shutil.copy2(settings_source, settings_target)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def record_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["id"].encode())
        digest.update(b"\0")
        digest.update(record["text"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    main()
