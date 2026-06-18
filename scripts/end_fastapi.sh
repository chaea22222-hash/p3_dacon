#!/bin/bash
# FastAPI tmux 세션을 종료한다.

set -e

SESSION="fastapi"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "FastAPI server is not running (no tmux session: '$SESSION')"
    exit 0
fi

tmux kill-session -t "$SESSION"
echo "FastAPI server stopped. (tmux session '$SESSION' terminated)"
