"""检查论文、表格和矢量图中的小数是否缺少前导 0。"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {".tex", ".md", ".csv", ".tsv", ".svg"}
MISSING_LEADING_ZERO = re.compile(r"(?<![\w\\])[-+]?\.\d+")


@dataclass(frozen=True)
class Violation:
    path: Path
    line_number: int
    value: str
    line: str


def iter_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path
        elif path.is_dir():
            yield from (
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
            )


def find_violations(paths: Iterable[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_text_files(paths):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in MISSING_LEADING_ZERO.finditer(line):
                violations.append(
                    Violation(
                        path=path,
                        line_number=line_number,
                        value=match.group(0),
                        line=line.strip(),
                    )
                )
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    violations = find_violations(args.paths)
    for violation in violations:
        print(
            f"{violation.path}:{violation.line_number}: "
            f"缺少前导 0：{violation.value} | {violation.line}"
        )
    if violations:
        print(f"共发现 {len(violations)} 处数字格式错误。")
        return 1
    print("数字格式检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
