"""CLI for running GenAIL scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from genai_lang.parser import ParseError, parse_script
from genai_lang.runtime import Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GenAIL scripts.")
    parser.add_argument("script", type=Path, help="Path to .gai script")
    args = parser.parse_args()

    try:
        source = args.script.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading script: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        statements = parse_script(source)
        runtime = Runtime()
        output = runtime.run(statements)
    except (ParseError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for line in output:
        print(line)


if __name__ == "__main__":
    main()
