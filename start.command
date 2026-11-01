#!/bin/bash
# 맥에서 더블클릭하면 필요한 걸 설치(처음 한 번)하고 분석 앱을 연다.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "처음 한 번만 설치합니다. 10분쯤 걸릴 수 있어요. 창을 닫지 마세요..."
  python3 -m venv .venv || { echo "Python 3가 필요합니다. https://www.python.org/downloads/ 에서 설치한 뒤 다시 열어 주세요."; read -n 1; exit 1; }
  .venv/bin/pip install -q -U pip && .venv/bin/pip install -r requirements.txt || { echo "설치에 실패했습니다. 인터넷 연결을 확인한 뒤 다시 열어 주세요."; rm -rf .venv; read -n 1; exit 1; }
fi
echo "앱을 여는 중... 잠시 후 브라우저가 열립니다. (끝내려면 이 창을 닫으세요)"
.venv/bin/python app.py
