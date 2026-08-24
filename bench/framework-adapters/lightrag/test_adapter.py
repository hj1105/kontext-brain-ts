import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from adapter import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FRAMEWORK_ID,
    PINNED_VERSION,
    normalize_context,
    prepare_warm_runtime,
    read_retrieval_checkpoint,
    record_digest,
    validate_warm_index,
)


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
