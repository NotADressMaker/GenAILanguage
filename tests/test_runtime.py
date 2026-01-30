import unittest

from genai_lang.parser import parse_script
from genai_lang.runtime import Runtime


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


if __name__ == "__main__":
    unittest.main()
