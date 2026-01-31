# GenAI Language (GenAIL)

GenAIL is a small, readable programming language for building **generative AI applications**.
It focuses on describing model configuration, prompts, tools, and chains in a compact script,
then executing those scripts with a pluggable runtime.

## Why GenAIL?

- **Simple:** line‑oriented syntax with minimal ceremony.
- **Composable:** define variables, build templates, and chain generations.
- **Extensible:** swap out the model provider or register custom tools.

## Install

```bash
python -m pip install -e .
```

## Running a script

```bash
genail examples/hello.gai
# or
python -m genai_lang.cli examples/hello.gai
```

## Language overview

### Statements

```text
# comments start with #
model "gpt-4o-mini"
set topic = "reinforcement learning"

message system "You are a concise research assistant."
message user "Explain {topic} in 2 sentences."

prompt """
Write a short summary about {topic}.
"""

generate summary from messages temperature=0.7 max_tokens=120
print summary
```

### Syntax reference

| Statement | Example | Description |
| --- | --- | --- |
| `model` | `model "gpt-4o-mini"` | Sets the active model name. |
| `set` | `set name = "Ada"` | Assigns a variable. |
| `template` | `template greeting = "Hello {name}!"` | Creates a formatted string using current variables. |
| `prompt` | `prompt """..."""` | Defines a multi‑line prompt. Stored as variable `prompt`. |
| `message` | `message user "Summarize {topic}."` | Appends a chat message to the `messages` list. |
| `generate` | `generate out from prompt temperature=0.7 max_tokens=120 format=json schema="{\"name\": \"\"}"` | Calls the model provider. |
| `call` | `call tool_name arg="value" into result` | Executes a tool and stores the result. |
| `print` | `print out` | Prints a variable or string literal. |

## Example

`examples/hello.gai`:

```text
model "demo-model"
set topic = "transformers"

prompt """
Create a short tagline for {topic}.
"""

generate tagline from prompt temperature=0.4 max_tokens=32
print tagline
```

`message` statements can build chat-style prompts:

```text
model "demo-model"
set topic = "tooling"
message system "You are a creative product assistant."
message user "List two app ideas about {topic}."
generate ideas from messages max_tokens=64
print ideas
```

## Extending the runtime

Provide your own model provider or tool registry:

```python
from genai_lang.runtime import Provider, Runtime

class MyProvider(Provider):
    def generate(self, model, prompt, temperature, max_tokens):
        return f"[{model}] {prompt.strip()}"

runtime = Runtime(provider=MyProvider())
```

Registering tools:

```python
from genai_lang.runtime import Runtime

runtime = Runtime()

runtime.register_tool("lookup", lambda key: {"key": key})
```

## Testing

```bash
pytest
```

## Project layout

- `genai_lang/parser.py`: Parser for the GenAIL syntax.
- `genai_lang/runtime.py`: Execution engine and provider interfaces.
- `genai_lang/cli.py`: CLI entry point.
- `examples/`: Sample scripts.
- `tests/`: Test suite.

## License

MIT
