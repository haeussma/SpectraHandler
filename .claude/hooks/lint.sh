#!/bin/sh
# PostToolUse: format + autofix the Python file that was just edited.
# Exit 2 hands anything ruff could not fix back to Claude.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
uv run --quiet ruff format -q "$f"
out=$(uv run --quiet ruff check --fix "$f" 2>&1) || { printf '%s\n' "$out" >&2; exit 2; }
