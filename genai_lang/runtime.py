"""Runtime for executing GenAIL scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Protocol

from genai_lang.parser import Statement


class Provider(Protocol):
    def generate(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Return generated text for a prompt."""

    def generate_messages(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return generated text for a chat message list."""


@dataclass
class MockProvider:
    """Default provider for local testing."""

    def generate(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        trimmed = " ".join(prompt.split())
        return (
            f"[{model}] ({temperature}, {max_tokens}) "
            f"{trimmed[:max_tokens].strip()}"
        )

    def generate_messages(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        prompt = Runtime.format_messages(messages)
        return self.generate(model, prompt, temperature, max_tokens)


@dataclass
class Runtime:
    provider: Provider = field(default_factory=MockProvider)
    tools: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    model: str = "default-model"
    output: List[str] = field(default_factory=list)

    def register_tool(self, name: str, handler: Any) -> None:
        self.tools[name] = handler

    @staticmethod
    def format_messages(messages: List[Dict[str, str]]) -> str:
        formatted: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def run(self, statements: Iterable[Statement]) -> List[str]:
        for stmt in statements:
            handler = getattr(self, f"_handle_{stmt.kind}", None)
            if not handler:
                raise ValueError(f"No handler for statement {stmt.kind}")
            handler(stmt.data)
        return self.output

    def _handle_model(self, data: Dict[str, Any]) -> None:
        self.model = str(data["name"])

    def _handle_set(self, data: Dict[str, Any]) -> None:
        self.variables[data["name"]] = data["value"]

    def _handle_template(self, data: Dict[str, Any]) -> None:
        template = str(data["value"])
        self.variables[data["name"]] = template.format(**self.variables)

    def _handle_prompt(self, data: Dict[str, Any]) -> None:
        self.variables["prompt"] = data["value"].format(**self.variables)

    def _handle_message(self, data: Dict[str, Any]) -> None:
        message = {
            "role": data["role"],
            "content": str(data["value"]).format(**self.variables),
        }
        messages = self.variables.setdefault("messages", [])
        if not isinstance(messages, list):
            raise ValueError("messages variable must be a list")
        messages.append(message)

    def _handle_generate(self, data: Dict[str, Any]) -> None:
        prompt_name = data["prompt"]
        prompt_value = self.variables.get(prompt_name, "")
        prompt_text = str(prompt_value)
        messages: List[Dict[str, str]] | None = None
        if isinstance(prompt_value, list):
            messages = prompt_value
            prompt_text = self.format_messages(messages)
        schema = data.get("schema")
        format_name = data.get("format")
        if schema:
            prompt_text = f"{prompt_text}\n\nReturn JSON matching this schema:\n{schema}"
        elif format_name == "json":
            prompt_text = f"{prompt_text}\n\nReturn valid JSON."
        if messages and hasattr(self.provider, "generate_messages"):
            generated = self.provider.generate_messages(
                self.model,
                messages,
                float(data["temperature"]),
                int(data["max_tokens"]),
            )
        else:
            generated = self.provider.generate(
                self.model,
                prompt_text,
                float(data["temperature"]),
                int(data["max_tokens"]),
            )
        if format_name == "json" or schema:
            try:
                generated_value = json.loads(generated)
            except json.JSONDecodeError as exc:
                raise ValueError("Generated output is not valid JSON") from exc
            self.variables[data["target"]] = generated_value
            return
        self.variables[data["target"]] = generated

    def _handle_call(self, data: Dict[str, Any]) -> None:
        tool_name = data["tool"]
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")
        result = tool(**data["args"])
        self.variables[data["target"]] = result

    def _handle_print(self, data: Dict[str, Any]) -> None:
        value = data["value"]
        if isinstance(value, str) and value in self.variables:
            value = self.variables[value]
        self.output.append(str(value))
