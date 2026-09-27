import tempfile
import unittest
from pathlib import Path

from research.path_consistent_negative_learning.scripts.lint_numeric_format import (
    find_violations,
)


class NumericLintTest(unittest.TestCase):
    def test_detects_missing_leading_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.tex"
            path.write_text("$p=.555$ and $\\rho=-.20$\n", encoding="utf-8")

            violations = find_violations([path])

        self.assertEqual([item.value for item in violations], [".555", "-.20"])

    def test_accepts_formatted_decimals_and_version_like_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "main.md"
            path.write_text(
                "0.00 0.10 0.25 0.50 0.75 1.00 p=0.555 rho=-0.20 v.1\n",
                encoding="utf-8",
            )

            violations = find_violations([path])

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
