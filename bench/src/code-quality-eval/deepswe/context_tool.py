#!/usr/bin/env python3
"""Dependency-free, arm-stable context command installed in DeepSWE agents."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def tokenize(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]+", value)}


def score(query: str, value: str) -> int:
    query_tokens = tokenize(query)
    value_tokens = tokenize(value)
    return len(query_tokens & value_tokens)


def snippet(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def ranked_documents(bundle: dict[str, Any], query: str, limit: int) -> list[dict[str, Any]]:
    scored = [
        (
            score(query, f"{document.get('title', '')}\n{document.get('body', '')}"),
            document,
        )
        for document in bundle.get("documents", [])
    ]
    ranked = [
        document
        for relevance, document in sorted(
            scored, key=lambda entry: (-entry[0], entry[1].get("documentId", ""))
        )
        if relevance > 0
    ]
    return [
        {
            "kind": "source",
            "documentId": document["documentId"],
            "title": document["title"],
            "sourceUri": document["sourceUri"],
            "observedAt": document["observedAt"],
            "ontologyNodeIds": document.get("ontologyNodeIds", []),
            "text": snippet(document["body"]),
        }
        for document in ranked[:limit]
    ]


def selector_score(record: dict[str, Any], path: str, symbol: str) -> int:
    total = 0
    for selector in record.get("symbolSelectors", []):
        relative_path = selector.get("relativePath")
        qualified_name = selector.get("qualifiedName")
        if relative_path and relative_path == path:
            total += 10
        if qualified_name and qualified_name == symbol:
            total += 20
    return total


def ranked_records(
    bundle: dict[str, Any], query: str, path: str, symbol: str, limit: int
) -> list[dict[str, Any]]:
    scored = [
        (score(query, record.get("text", "")) + selector_score(record, path, symbol), record)
        for record in bundle.get("normativeRecords", [])
    ]
    ranked = [
        record
        for relevance, record in sorted(
            scored, key=lambda entry: (-entry[0], entry[1].get("recordId", ""))
        )
        if relevance > 0
    ]
    return [
        {
            "kind": record["kind"],
            "recordId": record["recordId"],
            "revisionId": record["revisionId"],
            "text": record["text"],
            "evidenceIds": record.get("evidenceIds", []),
            "ontologyNodeIds": record.get("ontologyNodeIds", []),
        }
        for record in ranked[:limit]
    ]


def evidence_documents(
    bundle: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evidence_ids = {
        evidence_id
        for record in records
        for evidence_id in record.get("evidenceIds", [])
    }
    by_id = {
        document.get("documentId"): document
        for document in bundle.get("documents", [])
    }
    return [
        {
            "kind": "source",
            "documentId": document["documentId"],
            "title": document["title"],
            "sourceUri": document["sourceUri"],
            "observedAt": document["observedAt"],
            "ontologyNodeIds": document.get("ontologyNodeIds", []),
            "text": snippet(document["body"]),
        }
        for evidence_id in sorted(evidence_ids)
        if (document := by_id.get(evidence_id)) is not None
    ]


def log_call(command: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
    log_path = Path(os.environ.get("KONTEXT_EVAL_LOG", "/logs/agent/kontext-calls.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "observedAt": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "arguments": arguments,
        "ok": result.get("ok", False),
        "arm": result.get("arm"),
        "taskId": result.get("taskId"),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def load_bundle() -> dict[str, Any]:
    bundle_path = Path(os.environ.get("KONTEXT_EVAL_BUNDLE", "/opt/kontext-eval/bundle.json"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schemaVersion") != 1:
        raise ValueError("unsupported context bundle schema")
    return bundle


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kontext-context")
    subparsers = root.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-task")
    prepare.add_argument("--task-id", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5)
    begin = subparsers.add_parser("begin-logic")
    begin.add_argument("--path", required=True)
    begin.add_argument("--symbol", required=True)
    begin.add_argument("--responsibility", default="")
    begin.add_argument("--limit", type=int, default=8)
    check = subparsers.add_parser("check-change")
    check.add_argument("--path", required=True)
    check.add_argument("--symbol", required=True)
    check.add_argument("--tier", choices=("fast", "targeted"), required=True)
    return root


def execute(bundle: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    common = {
        "ok": True,
        "arm": bundle["arm"],
        "taskId": bundle["taskId"],
        "corpusSha256": bundle["corpusSha256"],
        "projectionSha256": bundle["projectionSha256"],
    }
    if args.command == "prepare-task":
        if args.task_id != bundle["taskId"]:
            return {**common, "ok": False, "error": "task id does not match context bundle"}
        return {**common, "contextAvailable": bundle["arm"] != "baseline"}
    if args.command == "search":
        limit = max(1, min(args.limit, 20))
        if bundle["arm"] == "baseline":
            return {**common, "results": []}
        if bundle["arm"] == "rag":
            return {**common, "results": ranked_documents(bundle, args.query, limit)}
        records = ranked_records(bundle, args.query, "", "", limit)
        return {
            **common,
            "results": records,
            "evidence": evidence_documents(bundle, records),
        }
    if args.command == "begin-logic":
        query = " ".join((args.path, args.symbol, args.responsibility))
        records = (
            ranked_records(bundle, query, args.path, args.symbol, args.limit)
            if bundle["arm"] == "kontext"
            else []
        )
        sources = (
            evidence_documents(bundle, records)
            if records
            else (
                []
                if bundle["arm"] == "baseline"
                else ranked_documents(bundle, query, args.limit)
            )
        )
        return {
            **common,
            "editingAllowed": True,
            "receipt": {
                "relativePath": args.path,
                "qualifiedName": args.symbol,
                "mandatoryRecords": records,
                "sources": sources,
            },
        }
    if args.command == "check-change":
        return {
            **common,
            "checked": True,
            "tier": args.tier,
            "relativePath": args.path,
            "qualifiedName": args.symbol,
        }
    raise AssertionError(f"unsupported command: {args.command}")


def main() -> int:
    args = parser().parse_args()
    bundle = load_bundle()
    result = execute(bundle, args)
    log_call(args.command, vars(args), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
