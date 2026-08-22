import unittest

from adapter import normalize_context


class NormalizeContextTest(unittest.TestCase):
    def test_treats_no_native_context_as_an_empty_success(self) -> None:
        self.assertEqual(normalize_context(None), "")

    def test_rejects_unexpected_non_string_context(self) -> None:
        with self.assertRaisesRegex(TypeError, "non-string context: dict"):
            normalize_context({})


if __name__ == "__main__":
    unittest.main()
