#!/bin/sh
# Stop: type-check and test once per turn (not per edit).
cd "$CLAUDE_PROJECT_DIR" || exit 0
fail=""
out=$(uv run --quiet ty check 2>&1) || fail="$out"
tout=$(uv run --quiet pytest 2>&1); c=$?
# pytest exit 5 == no tests collected, which is not a failure.
[ "$c" -eq 0 ] || [ "$c" -eq 5 ] || fail="$fail
$tout"
[ -z "$fail" ] || { printf '%s\n' "$fail" >&2; exit 2; }
