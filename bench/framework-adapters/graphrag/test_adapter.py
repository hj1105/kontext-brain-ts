import json
import tempfile
import unittest
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
    prepare_warm_runtime,
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
    def test_maps_system_and_chat_messages_onto_the_anthropic_request(self) -> None:
        client = FakeAnthropicClient(fake_anthropic_response(texts=("hello ", "world")))
        with mock.patch.object(adapter, "_anthropic_client", client):
            text = anthropic_complete_sync(
                [
                    {"role": "system", "content": "first rule"},
                    {"role": "system", "content": [{"type": "text", "text": "second rule"}]},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "draft"},
                    {"role": "user", "content": "refine"},
                ],
                "claude-sonnet-5",
                "medium",
                {},
            )

        self.assertEqual(text, "hello world")
        request = client.messages.requests[0]
        self.assertEqual(request["model"], "claude-sonnet-5")
        self.assertEqual(request["max_tokens"], 16000)
        self.assertEqual(request["output_config"], {"effort": "medium"})
        self.assertEqual(request["system"], "first rule\nsecond rule")
        self.assertEqual(request["messages"], [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "draft"},
            {"role": "user", "content": "refine"},
        ])
        self.assertNotIn("temperature", request)
        self.assertNotIn("thinking", request)

    def test_raises_on_refusal_and_max_tokens_stop_reasons(self) -> None:
        for stop_reason in ("refusal", "max_tokens"):
            with self.subTest(stop_reason=stop_reason):
                client = FakeAnthropicClient(fake_anthropic_response(stop_reason=stop_reason))
                with mock.patch.object(adapter, "_anthropic_client", client):
                    with self.assertRaisesRegex(RuntimeError, stop_reason):
                        anthropic_complete_sync(
                            [{"role": "user", "content": "question"}],
                            "claude-sonnet-5",
                            "medium",
                            {},
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


class WarmIndexTest(unittest.TestCase):
    def test_validates_marker_and_links_completed_output(self) -> None:
        corpus = [{"id": "doc-1", "text": "body"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            runtime = root / "runtime"
            (source / "output").mkdir(parents=True)
            (source / "settings.yaml").write_text("models: {}\n")
            (source / "rag-eval-build.json").write_text(json.dumps({
                "framework": FRAMEWORK_ID,
                "version": PINNED_VERSION,
                "corpusDigest": record_digest(corpus),
                "embeddingModel": EMBEDDING_MODEL,
                "embeddingDimensions": EMBEDDING_DIMENSIONS,
            }))

            validate_warm_index(source, corpus)
            prepare_warm_runtime(source, runtime)

            self.assertTrue((runtime / "output").is_symlink())
            self.assertFalse((runtime / "settings.yaml").is_symlink())

    def test_fails_closed_when_the_marker_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "warm index marker missing"):
                validate_warm_index(Path(directory), [{"id": "doc-1", "text": "body"}])


if __name__ == "__main__":
    unittest.main()
