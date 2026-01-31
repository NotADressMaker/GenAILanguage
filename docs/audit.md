# GenAIL Audit Notes

## Current State
- Script parsing, runtime, and CLI are implemented in a small set of modules.
- Tests exist (parser/runtime/golden) but focus on happy paths.
- Packaging metadata and CLI entry points are not configured.

## Initial Issues
- No pyproject.toml or packaging metadata for `pip install -e .` workflows.
- CLI does not surface parse/runtime errors with helpful exit codes.
- Parser and runtime error cases (missing variables, invalid arguments, unterminated prompts) are not covered by tests.
- Development tooling (pytest config, linting, CI) is missing.
