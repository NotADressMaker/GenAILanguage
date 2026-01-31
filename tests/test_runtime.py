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

    def test_generates_from_messages(self):
        class ChatProvider(Provider):
            def generate(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
                return "fallback"

            def generate_messages(
                self,
                model: str,
                messages: list[dict[str, str]],
                temperature: float,
                max_tokens: int,
            ) -> str:
                return f"{model}:{messages[-1]['content']}"

        script = """
message system "You are concise."
message user "Hi there."
generate reply from messages
"""
        statements = parse_script(script)
        runtime = Runtime(provider=ChatProvider())
        runtime.run(statements)
        self.assertEqual(runtime.variables["reply"], "default-model:Hi there.")

    def test_raises_on_missing_template_variable(self):
        script = """
template greeting = "Hello {topic}!"
"""
        statements = parse_script(script)
        runtime = Runtime()
        with self.assertRaises(ValueError):
            runtime.run(statements)

    def test_raises_on_reserved_prompt_assignment(self):
        script = """
set prompt = "override"
"""
        statements = parse_script(script)
        runtime = Runtime()
        with self.assertRaises(ValueError):
            runtime.run(statements)

    def test_raises_on_missing_prompt_variable(self):
        script = """
generate summary from prompt
"""
        statements = parse_script(script)
        runtime = Runtime()
        with self.assertRaises(ValueError):
            runtime.run(statements)

    def test_raises_on_invalid_generate_args(self):
        script = """
prompt \"\"\"
Hello
\"\"\"
generate result from prompt temperature=hot max_tokens=oops
"""
        statements = parse_script(script)
        runtime = Runtime()
        with self.assertRaises(ValueError):
            runtime.run(statements)


if __name__ == "__main__":
    unittest.main()
