import json
import tempfile
import unittest
from pathlib import Path

from adapter import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    FRAMEWORK_ID,
    PINNED_VERSION,
    prepare_warm_runtime,
    record_digest,
    validate_warm_index,
)


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
