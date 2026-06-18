#!/bin/bash
# MLflow tmux 세션을 종료한다.

set -e

SESSION="mlflow"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "MLflow server is not running (no tmux session: '$SESSION')"
    exit 0
fi

tmux kill-session -t "$SESSION"
echo "MLflow server stopped. (tmux session '$SESSION' terminated)"
