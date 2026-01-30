import unittest

from genai_lang.parser import ParseError, parse_script


class ParserTests(unittest.TestCase):
    def test_parses_prompt_and_generate(self):
        script = '''
model "demo"
set topic = "lang"

prompt """
Write about {topic}.
"""

generate out from prompt temperature=0.3 max_tokens=10
print out
'''
        statements = parse_script(script)
        kinds = [stmt.kind for stmt in statements]
        self.assertEqual(
            kinds,
            ["model", "set", "prompt", "generate", "print"],
        )

    def test_raises_on_bad_statement(self):
        with self.assertRaises(ParseError):
            parse_script("bad stuff")


if __name__ == "__main__":
    unittest.main()
