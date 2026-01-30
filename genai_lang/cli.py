"""CLI for running GenAIL scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from genai_lang.parser import parse_script
from genai_lang.runtime import Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GenAIL scripts.")
    parser.add_argument("script", type=Path, help="Path to .gai script")
    args = parser.parse_args()

    source = args.script.read_text(encoding="utf-8")
    statements = parse_script(source)
    runtime = Runtime()
    output = runtime.run(statements)

    for line in output:
        print(line)


if __name__ == "__main__":
    main()
