#!/bin/bash
# PostToolUse hook: triggers Axe code-reviewer on git diff after Edit/Write
# Runs only if there are staged or unstaged changes

DOCKER_SOCKET="${HOME}/.colima/default/docker.sock"
AXE_CONFIG="${HOME}/.config/axe"

# Check if docker is available
if [ ! -S "$DOCKER_SOCKET" ]; then
  exit 0
fi

# Get the diff (unstaged changes)
DIFF=$(git diff 2>/dev/null)

# Skip if no diff
if [ -z "$DIFF" ]; then
  exit 0
fi

# Only review if diff is meaningful (more than 5 lines changed)
DIFF_LINES=$(echo "$DIFF" | grep -c '^[+-]')
if [ "$DIFF_LINES" -lt 5 ]; then
  exit 0
fi

# Run Axe code-reviewer via Docker
REVIEW=$(echo "$DIFF" | docker -H "unix://${DOCKER_SOCKET}" run --rm -i \
  -v "${AXE_CONFIG}:/home/axe/.config/axe:ro" \
  axe run code-reviewer 2>/dev/null)

# Return review as additional context if non-empty
if [ -n "$REVIEW" ]; then
  echo "{\"additionalContext\": \"[AXE REVIEW] ${REVIEW}\"}"
fi
