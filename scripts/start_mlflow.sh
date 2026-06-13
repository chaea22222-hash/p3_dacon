#!/bin/bash
# MLflow 서버를 tmux 세션으로 시작한다.
# 이미 실행 중이면 메시지만 출력하고 종료한다.

set -e

SESSION="mlflow"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_BIN="$PROJECT_ROOT/.venv/bin/mlflow"

if [ ! -f "$MLFLOW_BIN" ]; then
    echo "Error: $MLFLOW_BIN not found. Run 'uv sync' first."
    exit 1
fi

SERVER_IP=$(hostname -I | awk '{print $1}')

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "MLflow server is already running (tmux session: '$SESSION')"
    echo "  View logs : tmux attach -t $SESSION"
    echo "  Dashboard : http://$SERVER_IP:5000"
    exit 0
fi

tmux new-session -d -s "$SESSION" \
    "$MLFLOW_BIN server \
    --backend-store-uri sqlite:///$HOME/dacon_project/mlflow.db \
    --default-artifact-root $HOME/dacon_project/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000"

sleep 2

if curl -s http://localhost:5000/health | grep -q "OK"; then
    echo "MLflow server started successfully."
    echo "  View logs : tmux attach -t $SESSION"
    echo "  Dashboard : http://$SERVER_IP:5000"
else
    echo "MLflow server may have failed to start."
    echo "  Check logs: tmux attach -t $SESSION"
    exit 1
fi
