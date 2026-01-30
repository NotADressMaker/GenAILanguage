import json
import pathlib
import unittest

from genai_lang.parser import parse_script
from genai_lang.runtime import Runtime

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden"


def _pretty_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class GoldenTests(unittest.TestCase):
    def test_parser_golden(self):
        script = (GOLDEN_DIR / "parser_script.gal").read_text()
        expected = json.loads((GOLDEN_DIR / "parser_expected.json").read_text())

        statements = parse_script(script)
        actual = [
            {
                "kind": stmt.kind,
                "data": stmt.data,
            }
            for stmt in statements
        ]

        self.assertMultiLineEqual(_pretty_json(actual), _pretty_json(expected))

    def test_runtime_golden(self):
        script = (GOLDEN_DIR / "runtime_script.gal").read_text()
        expected = json.loads((GOLDEN_DIR / "runtime_expected.json").read_text())

        statements = parse_script(script)
        runtime = Runtime()
        output = runtime.run(statements)

        actual = {
            "output": output,
            "variables": {
                "topic": runtime.variables.get("topic"),
                "greeting": runtime.variables.get("greeting"),
                "prompt": runtime.variables.get("prompt"),
                "summary": runtime.variables.get("summary"),
            },
        }

        self.assertMultiLineEqual(_pretty_json(actual), _pretty_json(expected))


if __name__ == "__main__":
    unittest.main()
