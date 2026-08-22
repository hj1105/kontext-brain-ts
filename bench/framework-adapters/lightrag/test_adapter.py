import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from adapter import normalize_context, read_retrieval_checkpoint


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


if __name__ == "__main__":
    unittest.main()
