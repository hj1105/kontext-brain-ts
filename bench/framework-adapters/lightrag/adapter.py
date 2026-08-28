#!/usr/bin/env python3
"""Official-default LightRAG 1.5.6 adapter for the RAG evaluation v2 contract."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai_embeddings import (  # noqa: E402
    OpenAIEmbeddingClient,
    codex_cli_environment,
    codex_evaluator_isolation_args,
)

import numpy as np
from lightrag import LightRAG, QueryParam, __version__ as LIGHTRAG_VERSION
from lightrag.utils import wrap_embedding_func_with_attrs


PINNED_VERSION = "1.5.6"
FRAMEWORK_ID = "lightrag"
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
        asyncio.run(build(args))
    else:
        asyncio.run(retrieve(args))


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
    issues: list[str] = []
    if LIGHTRAG_VERSION != PINNED_VERSION:
        issues.append(f"expected lightrag-hku {PINNED_VERSION}, found {LIGHTRAG_VERSION}")
    issues.extend(completion_execution_issues(completion_execution))
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY is not set")
    print(json.dumps({
        "status": "blocked" if issues else "ready",
        "version": LIGHTRAG_VERSION,
        "detail": "; ".join(issues) if issues else "official LightRAG core defaults with native mix retrieval",
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


async def build(args: argparse.Namespace) -> None:
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
            and marker.get("completionModel") == args.completion_model
            and marker.get("completionReasoningEffort") == args.completion_reasoning_effort
            and marker.get("completionExecution") == args.completion_execution
        ):
            return
    # The index directory is owned by this adapter. A different corpus or model
    # must start from an empty native store; inserting into the old store leaves
    # removed documents, entities, and cached completions reachable.
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    rag = create_rag(index_dir, args)
    await rag.initialize_storages()
    try:
        documents = [f"title: {item['title']}\nsource_id: {item['sourceId']}\n\n{item['text']}" for item in corpus]
        ids = [item["id"] for item in corpus]
        paths = [item["sourceId"] for item in corpus]
        await rag.ainsert(documents, ids=ids, file_paths=paths)
        write_json_atomic(marker_path, {
            "framework": FRAMEWORK_ID,
            "version": PINNED_VERSION,
            "corpusDigest": digest,
            "documents": len(corpus),
            "embeddingModel": EMBEDDING_MODEL,
            "embeddingDimensions": EMBEDDING_DIMENSIONS,
            "embeddingMode": "symmetric OpenAI embedding",
            "completionModel": args.completion_model,
            "completionReasoningEffort": args.completion_reasoning_effort,
            "completionExecution": args.completion_execution,
        })
    finally:
        await rag.finalize_storages()


async def retrieve(args: argparse.Namespace) -> None:
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
    rag = create_rag(index_dir, args, enable_llm_cache=not bool(args.index_source_dir))
    await rag.initialize_storages()
    try:
        query_semaphore = asyncio.Semaphore(args.query_concurrency)
        checkpoint_directory = retrieval_checkpoint_directory(index_dir, queries, args)

        async def retrieve_one(index: int, query: dict[str, Any]) -> dict[str, Any]:
            checkpoint_path = checkpoint_directory / f"{index:06d}.json"
            checkpoint = read_retrieval_checkpoint(checkpoint_path, args, query["id"])
            if checkpoint is not None:
                return checkpoint
            async with query_semaphore:
                # Match the other adapters' service-latency boundary: queue
                # admission is excluded, native retrieval work is included.
                started = time.perf_counter()
                started_at = datetime.now(timezone.utc).isoformat()
                try:
                    context = await rag.aquery(
                        query["text"],
                        QueryParam(
                            mode="mix",
                            only_need_context=True,
                            top_k=args.top_k,
                            chunk_top_k=args.top_k,
                            enable_rerank=False,
                        ),
                    )
                    context = normalize_context(context)
                    evidence = [] if not context.strip() else [{
                        "id": f"{FRAMEWORK_ID}:{query['id']}:context",
                        "sourceId": "lightrag-context",
                        "text": context,
                        "score": 1,
                        "rank": 1,
                        "metadata": {"mode": "mix", "nativeContext": True},
                    }]
                    record = retrieval_record(
                        args, query["id"], "ok", evidence, started, started_at, None
                    )
                except Exception as error:  # preserve per-query failures
                    record = retrieval_record(
                        args, query["id"], "error", [], started, started_at, str(error)
                    )
                write_json_atomic(checkpoint_path, record)
                return record

        # asyncio.gather preserves input order; the harness freezes the worker
        # count explicitly so clean latency runs never inherit a package default.
        records = await asyncio.gather(*(
            retrieve_one(index, query) for index, query in enumerate(queries)
        ))
    finally:
        await rag.finalize_storages()
    write_jsonl(Path(args.output), records)


def normalize_context(context: Any) -> str:
    if context is None:
        return ""
    if not isinstance(context, str):
        raise TypeError(
            "LightRAG returned non-string context: "
            f"{type(context).__name__}"
        )
    return context


def create_rag(
    index_dir: Path,
    args: argparse.Namespace,
    *,
    enable_llm_cache: bool = True,
) -> LightRAG:
    embedding_client = OpenAIEmbeddingClient(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        usage_path=index_dir / "openai-embedding-usage.jsonl",
    )

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_DIMENSIONS,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
        supports_asymmetric=False,
    )
    async def embed(texts: list[str], context: str = "document") -> np.ndarray:
        del context
        vectors = await asyncio.to_thread(embedding_client.embed, texts)
        return np.asarray(vectors, dtype=np.float32)

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **_kwargs: Any,
    ) -> str:
        complete_one = (
            anthropic_complete
            if args.completion_execution == "anthropic-api"
            else codex_complete
        )
        return await complete_one(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            model=args.completion_model,
            reasoning_effort=args.completion_reasoning_effort,
        )

    return LightRAG(
        working_dir=str(index_dir),
        embedding_func=embed,
        llm_model_func=complete,
        llm_model_name=args.completion_model,
        enable_llm_cache=enable_llm_cache,
        auto_manage_storages_states=False,
    )


async def codex_complete(
    *,
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict[str, str]],
    model: str,
    reasoning_effort: str,
) -> str:
    return await asyncio.to_thread(
        codex_complete_sync,
        prompt,
        system_prompt,
        history_messages,
        model,
        reasoning_effort,
    )


def codex_complete_sync(
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict[str, str]],
    model: str,
    reasoning_effort: str,
) -> str:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": {"type": "string"}},
    }
    messages = "\n".join(
        f"<{item.get('role', 'user')}>\n{item.get('content', '')}\n</{item.get('role', 'user')}>"
        for item in history_messages
    )
    request = "\n".join([
        "Act as a deterministic completion backend for the embedded request.",
        "Do not use tools, files, web search, or prior conversation.",
        "Return the exact requested completion in the JSON field `text`.",
        "If the request asks for JSON, `text` must contain valid JSON without markdown fences.",
        f"<system>\n{system_prompt or ''}\n</system>",
        f"<history>\n{messages}\n</history>",
        f"<prompt>\n{prompt}\n</prompt>",
    ])
    with tempfile.TemporaryDirectory(prefix="rag-eval-lightrag-") as directory:
        schema_path = Path(directory) / "schema.json"
        output_path = Path(directory) / "output.json"
        schema_path.write_text(json.dumps(schema))
        result = subprocess.run([
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            *codex_evaluator_isolation_args(),
            "--skip-git-repo-check", "--sandbox", "read-only", "--json",
            "--model", model, "--config", f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema", str(schema_path), "--output-last-message", str(output_path), "-",
        ], input=request, text=True, capture_output=True, timeout=600,
            env=codex_cli_environment())
        if result.returncode != 0:
            raise RuntimeError(f"codex exec failed: {result.stderr}\n{result.stdout}")
        return json.loads(output_path.read_text())["text"]


async def anthropic_complete(
    *,
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict[str, str]],
    model: str,
    reasoning_effort: str,
) -> str:
    return await asyncio.to_thread(
        anthropic_complete_sync,
        prompt,
        system_prompt,
        history_messages,
        model,
        reasoning_effort,
    )


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
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Map LightRAG's prompt/system/history onto (system prompt, Anthropic messages)."""
    system_parts: list[str] = [system_prompt] if system_prompt else []
    chat: list[dict[str, str]] = []
    for item in history_messages:
        role = item.get("role", "user")
        content = str(item.get("content", ""))
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            chat.append({"role": role, "content": content})
        else:
            raise ValueError(f"unsupported chat message role: {role!r}")
    chat.append({"role": "user", "content": prompt})
    return "\n".join(system_parts), chat


def anthropic_complete_sync(
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict[str, str]],
    model: str,
    reasoning_effort: str,
) -> str:
    system, chat = anthropic_request_messages(prompt, system_prompt, history_messages)
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
    return "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )


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
    """Link immutable native stores into a clean, separately writable runtime."""
    if index_source_dir == index_dir.resolve():
        raise RuntimeError("warm index source and clean runtime must be different directories")
    index_dir.mkdir(parents=True, exist_ok=True)
    excluded = {
        "kv_store_llm_response_cache.json",
        "openai-embedding-usage.jsonl",
        "retrieval-checkpoints",
    }
    for source in index_source_dir.iterdir():
        if source.name in excluded:
            continue
        target = index_dir / source.name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise RuntimeError(f"clean runtime collision: {target}")
            continue
        target.symlink_to(source, target_is_directory=source.is_dir())


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


def retrieval_checkpoint_directory(
    index_dir: Path,
    queries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    digest = hashlib.sha256()
    digest.update(record_digest(queries).encode())
    marker_path = index_dir / "rag-eval-build.json"
    if not marker_path.exists():
        raise RuntimeError("LightRAG index marker is missing; run build before retrieve")
    marker = json.loads(marker_path.read_text())
    digest.update(b"\0")
    digest.update(json.dumps(marker, sort_keys=True, separators=(",", ":")).encode())
    for value in (
        PINNED_VERSION,
        args.embedding_model,
        str(args.embedding_dimensions),
        args.completion_model,
        args.completion_reasoning_effort,
        args.completion_execution,
        str(args.top_k),
        "mix",
        "rerank=false",
    ):
        digest.update(b"\0")
        digest.update(value.encode())
    directory = index_dir / "retrieval-checkpoints" / digest.hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_retrieval_checkpoint(
    path: Path,
    args: argparse.Namespace,
    query_id: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("datasetId") != Path(args.dataset_dir).parents[1].name
        or record.get("frameworkId") != FRAMEWORK_ID
        or record.get("queryId") != query_id
        or record.get("status") != "ok"
    ):
        return None
    return record


def record_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for key in ("id", "sourceId", "title", "text"):
            digest.update(key.encode())
            digest.update(b"\0")
            digest.update(str(record.get(key, "")).encode())
            digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    main()
