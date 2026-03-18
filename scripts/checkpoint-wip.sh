#!/bin/bash
# WIP Persistence: PreCompact hook
# Context compact olmadan once mevcut calisma durumunu kaydeder

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
SESSION=$(echo "$INPUT" | jq -r '.session_id // empty')

[ -z "$CWD" ] && exit 0

STATE_DIR="$HOME/.claude/state"
mkdir -p "$STATE_DIR"

PROJECT_NAME=$(basename "$CWD")
CHECKPOINT="$STATE_DIR/wip-${PROJECT_NAME}.json"

# Git bilgileri
BRANCH=""
MODIFIED=""
STATUS=""
if [ -d "$CWD/.git" ] || git -C "$CWD" rev-parse --git-dir >/dev/null 2>&1; then
  BRANCH=$(cd "$CWD" && git branch --show-current 2>/dev/null || echo "")
  MODIFIED=$(cd "$CWD" && git diff --name-only 2>/dev/null | head -20 || echo "")
  STATUS=$(cd "$CWD" && git diff --stat 2>/dev/null | tail -1 || echo "")
fi

# Staged dosyalar
STAGED=""
if [ -d "$CWD/.git" ]; then
  STAGED=$(cd "$CWD" && git diff --cached --name-only 2>/dev/null | head -20 || echo "")
fi

# TodoWrite task dosyasi varsa oku
TODO_FILE="$HOME/.claude/todos/${PROJECT_NAME}.json"
TODOS=""
if [ -f "$TODO_FILE" ]; then
  TODOS=$(cat "$TODO_FILE" 2>/dev/null || echo "")
fi

jq -n \
  --arg cwd "$CWD" \
  --arg project "$PROJECT_NAME" \
  --arg session "$SESSION" \
  --arg branch "$BRANCH" \
  --arg modified "$MODIFIED" \
  --arg staged "$STAGED" \
  --arg status "$STATUS" \
  --arg todos "$TODOS" \
  --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    cwd: $cwd,
    project: $project,
    session: $session,
    branch: $branch,
    modified_files: ($modified | split("\n") | map(select(. != ""))),
    staged_files: ($staged | split("\n") | map(select(. != ""))),
    git_status: $status,
    todos: $todos,
    timestamp: $time
  }' > "$CHECKPOINT"

exit 0
