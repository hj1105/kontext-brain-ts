import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import adapter
from adapter import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FRAMEWORK_ID,
    PINNED_VERSION,
    anthropic_complete_sync,
    completion_execution_issues,
    normalize_context,
    prepare_warm_runtime,
    read_retrieval_checkpoint,
    record_digest,
    validate_warm_index,
)


class FakeAnthropicMessages:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.requests: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.messages = FakeAnthropicMessages(response)


def fake_anthropic_response(
    stop_reason: str = "end_turn", texts: tuple[str, ...] = ("hello",)
) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text) for text in texts],
    )


class AnthropicCompletionTest(unittest.TestCase):
    def test_maps_system_history_and_prompt_onto_the_anthropic_request(self) -> None:
        client = FakeAnthropicClient(fake_anthropic_response(texts=("hello ", "world")))
        with mock.patch.object(adapter, "_anthropic_client", client):
            text = anthropic_complete_sync(
                "final question",
                "be terse",
                [
                    {"role": "system", "content": "extra rule"},
                    {"role": "user", "content": "earlier question"},
                    {"role": "assistant", "content": "earlier answer"},
                ],
                "claude-sonnet-5",
                "medium",
            )

        self.assertEqual(text, "hello world")
        request = client.messages.requests[0]
        self.assertEqual(request["model"], "claude-sonnet-5")
        self.assertEqual(request["max_tokens"], 16000)
        self.assertEqual(request["output_config"], {"effort": "medium"})
        self.assertEqual(request["system"], "be terse\nextra rule")
        self.assertEqual(request["messages"], [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "final question"},
        ])
        self.assertNotIn("temperature", request)
        self.assertNotIn("thinking", request)

    def test_omits_the_system_parameter_when_no_system_content_exists(self) -> None:
        client = FakeAnthropicClient(fake_anthropic_response())
        with mock.patch.object(adapter, "_anthropic_client", client):
            anthropic_complete_sync("question", None, [], "claude-sonnet-5", "medium")

        self.assertNotIn("system", client.messages.requests[0])

    def test_raises_on_refusal_and_max_tokens_stop_reasons(self) -> None:
        for stop_reason in ("refusal", "max_tokens"):
            with self.subTest(stop_reason=stop_reason):
                client = FakeAnthropicClient(fake_anthropic_response(stop_reason=stop_reason))
                with mock.patch.object(adapter, "_anthropic_client", client):
                    with self.assertRaisesRegex(RuntimeError, stop_reason):
                        anthropic_complete_sync(
                            "question", None, [], "claude-sonnet-5", "medium"
                        )


class DoctorCompletionExecutionTest(unittest.TestCase):
    def test_codex_mode_checks_only_the_codex_cli(self) -> None:
        with mock.patch.object(adapter.shutil, "which", return_value=None):
            self.assertEqual(
                completion_execution_issues("codex-exec"), ["codex CLI is not on PATH"]
            )
        with mock.patch.object(adapter.shutil, "which", return_value="/usr/bin/codex"):
            self.assertEqual(completion_execution_issues("codex-exec"), [])

    def test_anthropic_mode_checks_key_and_package_instead_of_codex(self) -> None:
        with mock.patch.object(adapter.shutil, "which", return_value=None), \
                mock.patch.dict(adapter.os.environ, {"ANTHROPIC_API_KEY": ""}), \
                mock.patch.object(adapter, "anthropic_package_available", return_value=False):
            self.assertEqual(completion_execution_issues("anthropic-api"), [
                "ANTHROPIC_API_KEY is not set",
                "anthropic package is not importable",
            ])
        with mock.patch.object(adapter.shutil, "which", return_value=None), \
                mock.patch.dict(adapter.os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
                mock.patch.object(adapter, "anthropic_package_available", return_value=True):
            self.assertEqual(completion_execution_issues("anthropic-api"), [])


class NormalizeContextTest(unittest.TestCase):
    def test_treats_no_native_context_as_an_empty_success(self) -> None:
        self.assertEqual(normalize_context(None), "")

    def test_rejects_unexpected_non_string_context(self) -> None:
        with self.assertRaisesRegex(TypeError, "non-string context: dict"):
            normalize_context({})


class RetrievalCheckpointTest(unittest.TestCase):
    def test_reuses_only_successful_retrievals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "graphrag-bench-novel" / "lightrag" / "dataset"
            args = Namespace(dataset_dir=str(dataset_dir))
            checkpoint = root / "checkpoint.json"
            record = {
                "datasetId": "graphrag-bench-novel",
                "frameworkId": "lightrag",
                "queryId": "query-1",
                "status": "ok",
            }
            checkpoint.write_text(json.dumps(record))

            self.assertEqual(read_retrieval_checkpoint(checkpoint, args, "query-1"), record)

            checkpoint.write_text(json.dumps({**record, "status": "error"}))
            self.assertIsNone(read_retrieval_checkpoint(checkpoint, args, "query-1"))


class WarmIndexTest(unittest.TestCase):
    def test_validates_marker_and_keeps_query_cache_out_of_clean_runtime(self) -> None:
        corpus = [{"id": "doc-1", "text": "body"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            runtime = root / "runtime"
            source.mkdir()
            (source / "rag-eval-build.json").write_text(json.dumps({
                "framework": FRAMEWORK_ID,
                "version": PINNED_VERSION,
                "corpusDigest": record_digest(corpus),
                "embeddingModel": EMBEDDING_MODEL,
                "embeddingDimensions": EMBEDDING_DIMENSIONS,
            }))
            (source / "vdb_chunks.json").write_text("{}")
            (source / "kv_store_llm_response_cache.json").write_text("{\"cached\": true}")

            validate_warm_index(source, corpus)
            prepare_warm_runtime(source, runtime)

            self.assertTrue((runtime / "vdb_chunks.json").is_symlink())
            self.assertFalse((runtime / "kv_store_llm_response_cache.json").exists())

    def test_fails_closed_on_a_corpus_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "rag-eval-build.json").write_text(json.dumps({
                "framework": FRAMEWORK_ID,
                "version": PINNED_VERSION,
                "corpusDigest": "wrong",
                "embeddingModel": EMBEDDING_MODEL,
                "embeddingDimensions": EMBEDDING_DIMENSIONS,
            }))
            with self.assertRaisesRegex(RuntimeError, "warm index marker mismatch"):
                validate_warm_index(source, [{"id": "doc-1", "text": "body"}])


if __name__ == "__main__":
    unittest.main()
