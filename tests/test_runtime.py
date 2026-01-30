import unittest

from genai_lang.parser import parse_script
from genai_lang.runtime import Provider, Runtime


class RuntimeTests(unittest.TestCase):
    def test_runs_and_collects_output(self):
        script = """
model "demo"
set topic = "agents"

template greeting = "Hello {topic}!"
print greeting
"""
        statements = parse_script(script)
        runtime = Runtime()
        output = runtime.run(statements)
        self.assertEqual(output, ["Hello agents!"])

    def test_generates_json_output(self):
        class JsonProvider(Provider):
            def generate(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
                return "{\"name\": \"Ada\"}"

        script = """
prompt \"\"\"
Return JSON.
\"\"\"
generate result from prompt format=json
"""
        statements = parse_script(script)
        runtime = Runtime(provider=JsonProvider())
        runtime.run(statements)
        self.assertEqual(runtime.variables["result"], {"name": "Ada"})


if __name__ == "__main__":
    unittest.main()
