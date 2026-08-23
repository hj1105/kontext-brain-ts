import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path

# Keep pure adapter helper tests runnable without installing the heavyweight
# native LightRAG environment. The real package/version is still checked by the
# adapter's doctor command in its isolated uv project.
numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
numpy_stub.float32 = float
numpy_stub.asarray = lambda values, dtype=None: values
sys.modules.setdefault("numpy", numpy_stub)

lightrag_stub = types.ModuleType("lightrag")
lightrag_stub.LightRAG = object
lightrag_stub.QueryParam = object
lightrag_stub.__version__ = "1.5.6"
sys.modules.setdefault("lightrag", lightrag_stub)
lightrag_utils_stub = types.ModuleType("lightrag.utils")
lightrag_utils_stub.wrap_embedding_func_with_attrs = lambda **_kwargs: lambda function: function
sys.modules.setdefault("lightrag.utils", lightrag_utils_stub)

from adapter import (  # noqa: E402
    normalize_context,
    read_retrieval_checkpoint,
    record_digest,
    retrieval_checkpoint_directory,
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

    def test_checkpoint_key_covers_index_and_model_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            marker_path = index_dir / "rag-eval-build.json"
            marker_path.write_text(json.dumps({"corpusDigest": "corpus-a"}))
            args = Namespace(
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536,
                completion_model="gpt-5.6-terra",
                completion_reasoning_effort="medium",
                completion_execution="codex-exec",
                top_k=10,
            )
            queries = [{"id": "query-1", "text": "Question?"}]

            first = retrieval_checkpoint_directory(index_dir, queries, args)
            marker_path.write_text(json.dumps({"corpusDigest": "corpus-b"}))
            second = retrieval_checkpoint_directory(index_dir, queries, args)
            changed_model = Namespace(**{**vars(args), "completion_model": "different"})
            third = retrieval_checkpoint_directory(index_dir, queries, changed_model)

            self.assertNotEqual(first, second)
            self.assertNotEqual(second, third)

    def test_record_digest_includes_source_metadata(self) -> None:
        original = [{"id": "doc", "sourceId": "one", "title": "Title", "text": "Body"}]
        changed = [{**original[0], "sourceId": "two"}]

        self.assertNotEqual(record_digest(original), record_digest(changed))


if __name__ == "__main__":
    unittest.main()
