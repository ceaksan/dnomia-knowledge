#!/bin/bash
# WIP Persistence: SessionStart hook
# Onceki session'dan kalan checkpoint'i restore eder

INPUT=$(cat)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
SOURCE=$(echo "$INPUT" | jq -r '.source // empty')

[ -z "$CWD" ] && exit 0

PROJECT_NAME=$(basename "$CWD")
CHECKPOINT="$HOME/.claude/state/wip-${PROJECT_NAME}.json"

# Checkpoint yoksa veya 24 saatten eskiyse atla
if [ ! -f "$CHECKPOINT" ]; then
  exit 0
fi

# 24 saat kontrolu (86400 saniye)
if [ "$(uname)" = "Darwin" ]; then
  FILE_AGE=$(( $(date +%s) - $(stat -f %m "$CHECKPOINT") ))
else
  FILE_AGE=$(( $(date +%s) - $(stat -c %Y "$CHECKPOINT") ))
fi

if [ "$FILE_AGE" -gt 86400 ]; then
  rm -f "$CHECKPOINT"
  exit 0
fi

# Checkpoint'i oku ve context olustur
BRANCH=$(jq -r '.branch // ""' "$CHECKPOINT")
MODIFIED=$(jq -r '.modified_files | join(", ")' "$CHECKPOINT")
STAGED=$(jq -r '.staged_files | join(", ")' "$CHECKPOINT")
GIT_STATUS=$(jq -r '.git_status // ""' "$CHECKPOINT")
TODOS=$(jq -r '.todos // ""' "$CHECKPOINT")
REFLECTION=$(jq -r '.reflection // ""' "$CHECKPOINT")
TIMESTAMP=$(jq -r '.timestamp // ""' "$CHECKPOINT")

# Context mesaji olustur
CONTEXT="[WIP RECOVERED] Onceki session'dan kalan calisma durumu (${TIMESTAMP}):"
[ -n "$BRANCH" ] && CONTEXT="$CONTEXT\nBranch: $BRANCH"
[ -n "$MODIFIED" ] && [ "$MODIFIED" != "" ] && CONTEXT="$CONTEXT\nDegisen dosyalar: $MODIFIED"
[ -n "$STAGED" ] && [ "$STAGED" != "" ] && CONTEXT="$CONTEXT\nStaged: $STAGED"
[ -n "$GIT_STATUS" ] && CONTEXT="$CONTEXT\nGit durumu: $GIT_STATUS"
[ -n "$TODOS" ] && [ "$TODOS" != "" ] && CONTEXT="$CONTEXT\nAktif gorevler: $TODOS"
[ -n "$REFLECTION" ] && [ "$REFLECTION" != "" ] && CONTEXT="$CONTEXT\nSon aktivite: $REFLECTION"

# additionalContext olarak dondur
jq -n --arg ctx "$(echo -e "$CONTEXT")" '{additionalContext: $ctx}'

# Checkpoint'i temizle (bir kez restore edildi)
rm -f "$CHECKPOINT"

exit 0
