import pathlib
import unittest

from genai_lang.parser import parse_script
from genai_lang.runtime import Provider, Runtime


class DemoProvider(Provider):
    def generate(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        return f"[{model}] {prompt.strip()}"


class ExampleTests(unittest.TestCase):
    def test_hello_example(self):
        script_path = pathlib.Path(__file__).parent.parent / "examples" / "hello.gai"
        script = script_path.read_text(encoding="utf-8")
        statements = parse_script(script)
        runtime = Runtime(provider=DemoProvider())
        output = runtime.run(statements)
        self.assertEqual(
            output,
            ["[demo-model] Create a short tagline for transformers."],
        )


if __name__ == "__main__":
    unittest.main()
