#!/bin/bash
# FastAPI 예측 서버를 tmux 세션으로 시작한다.
# 이미 실행 중이면 메시지만 출력하고 종료한다.

set -e

SESSION="fastapi"
PORT=8151
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="$PROJECT_ROOT/.venv/bin/uvicorn"

if [ ! -f "$UV_BIN" ]; then
    echo "Error: $UV_BIN not found. Run 'uv sync' first."
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "FastAPI server is already running (tmux session: '$SESSION')"
    echo "  View logs : tmux attach -t $SESSION"
    echo "  Dashboard : http://localhost:$PORT/dashboard"
    exit 0
fi

if pgrep -f "uvicorn.*api.main" > /dev/null 2>&1; then
    echo "Error: FastAPI process already running outside of tmux."
    echo "  Stop first: pkill -f 'uvicorn.*api.main'"
    exit 1
fi

tmux new-session -d -s "$SESSION" \
    "cd '$PROJECT_ROOT' && uv run uvicorn api.main:app --host 127.0.0.1 --port $PORT"

echo "Waiting for FastAPI server to be ready..."
for i in $(seq 1 10); do
    if curl -s http://localhost:$PORT/health | grep -q "ok\|OK\|healthy\|status"; then
        echo "FastAPI server started successfully."
        echo "  View logs : tmux attach -t $SESSION"
        echo "  Dashboard : http://localhost:$PORT/dashboard"
        exit 0
    fi
    sleep 1
done

echo "FastAPI server may have failed to start."
echo "  Check logs: tmux attach -t $SESSION"
exit 1
