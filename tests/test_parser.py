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

    def test_parses_generate_json_args(self):
        script = """
prompt \"\"\"
Return info.
\"\"\"
generate result from prompt format=json schema=\"{\\\"name\\\": \\\"\\\"}\"
"""
        statements = parse_script(script)
        generate_stmt = next(stmt for stmt in statements if stmt.kind == "generate")
        self.assertEqual(generate_stmt.data["format"], "json")
        self.assertEqual(generate_stmt.data["schema"], "{\"name\": \"\"}")

    def test_parses_message_statement(self):
        script = """
message system "You are helpful."
"""
        statements = parse_script(script)
        self.assertEqual(statements[0].kind, "message")
        self.assertEqual(statements[0].data["role"], "system")
        self.assertEqual(statements[0].data["value"], "You are helpful.")


if __name__ == "__main__":
    unittest.main()
