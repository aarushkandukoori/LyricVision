#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Backend
cd "$ROOT/backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

# Frontend
cd "$ROOT/frontend"
npm install --silent
npm run build

# Serve
cd "$ROOT/backend"
echo ""
echo "LyricVision running at http://127.0.0.1:8000"
echo ""
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
