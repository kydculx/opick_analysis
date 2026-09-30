#!/bin/sh
# 콘솔 학습 데스크톱 GUI 실행
#   sh ml/train_gui.sh        # 백그라운드 실행 (콘솔 즉시 반환)
#   sh ml/train_gui.sh --fg   # 포그라운드 실행 (로그 직접 확인용)
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env.local 2>/dev/null || true; set +a
if [ "${1:-}" = "--fg" ]; then
  shift
  exec ml/.venv/bin/python ml/grid_console.py "$@"
fi
LOG="ml/.gui.log"
mkdir -p ml
nohup ml/.venv/bin/python ml/grid_console.py "$@" >"$LOG" 2>&1 &
echo "GUI 시작됨 (pid $!, 로그: $LOG)"
