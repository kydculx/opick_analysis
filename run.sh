#!/bin/sh
# 웹서버 재시작 (기존 종료 → 최신 빌드 → 실행)
#   sh run.sh            → 재빌드 후 8002로 재시작
#   sh run.sh --no-build → 빌드 생략, 재시작만
set -e
cd "$(dirname "$0")"
PORT="8002"
if [ "${1:-}" != "--no-build" ]; then
  npm run build
else
  shift
fi
if lsof -ti:$PORT >/dev/null 2>&1; then
  echo "기존 서버 종료 (port $PORT)"
  lsof -ti:$PORT | xargs kill -9
  sleep 2
fi
set -a; . ./.env.local 2>/dev/null || true; set +a
echo "서버 시작 (port $PORT)"
exec npm run start -- --port "$PORT" "$@"