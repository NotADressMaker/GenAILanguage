"""Parser for the GenAIL language."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class Statement:
    kind: str
    data: Dict[str, Any]


ASSIGN_RE = re.compile(r"^(?P<name>[a-zA-Z_][\w]*)\s*=\s*(?P<value>.+)$")
MODEL_RE = re.compile(r"^model\s+\"(?P<name>.+)\"$")
TEMPLATE_RE = re.compile(r"^template\s+(?P<name>[a-zA-Z_][\w]*)\s*=\s+\"(?P<value>.*)\"$")
SET_RE = re.compile(r"^set\s+(?P<rest>.+)$")
GENERATE_RE = re.compile(
    r"^generate\s+(?P<target>[a-zA-Z_][\w]*)\s+from\s+(?P<prompt>[a-zA-Z_][\w]*)"
    r"(?P<rest>.*)$"
)
CALL_RE = re.compile(
    r"^call\s+(?P<tool>[a-zA-Z_][\w]*)\s+(?P<args>.+?)\s+into\s+(?P<target>[a-zA-Z_][\w]*)$"
)
PRINT_RE = re.compile(r"^print\s+(?P<value>.+)$")
MESSAGE_RE = re.compile(
    r"^message\s+(?P<role>[a-zA-Z_][\w]*)\s+\"(?P<value>.*)\"$"
)
PROMPT_START_RE = re.compile(r"^prompt\s+\"\"\"\s*$")
PROMPT_END_RE = re.compile(r"^\"\"\"\s*$")


class ParseError(ValueError):
    """Raised when a script cannot be parsed."""


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("\"") and raw.endswith("\""):
        return raw[1:-1]
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw


def _parse_call_args(raw: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for pair in raw.split():
        match = ASSIGN_RE.match(pair)
        if not match:
            raise ParseError(f"Invalid call argument: {pair}")
        args[match.group("name")] = _parse_value(match.group("value"))
    return args


def _parse_generate_args(raw: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "temperature": 0.7,
        "max_tokens": 128,
        "format": None,
        "schema": None,
    }
    if not raw:
        return args
    tokens = shlex.split(raw)
    for token in tokens:
        match = ASSIGN_RE.match(token)
        if not match:
            raise ParseError(f"Invalid generate argument: {token}")
        key = match.group("name")
        value = _parse_value(match.group("value"))
        if key not in args:
            raise ParseError(f"Unsupported generate argument: {key}")
        args[key] = value
    return args


def _consume_prompt(lines: Iterable[str], start_index: int) -> tuple[str, int]:
    collected: List[str] = []
    for offset, line in enumerate(lines):
        if offset == 0:
            continue
        if PROMPT_END_RE.match(line):
            return "\n".join(collected), start_index + offset
        collected.append(line)
    raise ParseError("Unterminated prompt block")


def parse_script(source: str) -> List[Statement]:
    statements: List[Statement] = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index].rstrip()
        line = raw_line.strip()
        if not line or line.startswith("#"):
            index += 1
            continue

        if PROMPT_START_RE.match(line):
            prompt, end_index = _consume_prompt(lines[index:], index)
            statements.append(Statement("prompt", {"value": prompt}))
            index = end_index + 1
            continue

        model_match = MODEL_RE.match(line)
        if model_match:
            statements.append(Statement("model", {"name": model_match.group("name")}))
            index += 1
            continue

        template_match = TEMPLATE_RE.match(line)
        if template_match:
            statements.append(
                Statement(
                    "template",
                    {"name": template_match.group("name"), "value": template_match.group("value")},
                )
            )
            index += 1
            continue

        set_match = SET_RE.match(line)
        if set_match:
            assign_match = ASSIGN_RE.match(set_match.group("rest"))
            if not assign_match:
                raise ParseError(f"Invalid set statement: {line}")
            statements.append(
                Statement(
                    "set",
                    {
                        "name": assign_match.group("name"),
                        "value": _parse_value(assign_match.group("value")),
                    },
                )
            )
            index += 1
            continue

        generate_match = GENERATE_RE.match(line)
        if generate_match:
            args = _parse_generate_args(generate_match.group("rest").strip())
            statements.append(
                Statement(
                    "generate",
                    {
                        "target": generate_match.group("target"),
                        "prompt": generate_match.group("prompt"),
                        **args,
                    },
                )
            )
            index += 1
            continue

        call_match = CALL_RE.match(line)
        if call_match:
            statements.append(
                Statement(
                    "call",
                    {
                        "tool": call_match.group("tool"),
                        "args": _parse_call_args(call_match.group("args")),
                        "target": call_match.group("target"),
                    },
                )
            )
            index += 1
            continue

        message_match = MESSAGE_RE.match(line)
        if message_match:
            statements.append(
                Statement(
                    "message",
                    {
                        "role": message_match.group("role"),
                        "value": message_match.group("value"),
                    },
                )
            )
            index += 1
            continue

        print_match = PRINT_RE.match(line)
        if print_match:
            statements.append(
                Statement("print", {"value": _parse_value(print_match.group("value"))})
            )
            index += 1
            continue

        raise ParseError(f"Unrecognized statement: {line}")

    return statements
