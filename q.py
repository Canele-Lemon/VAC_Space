import os, sys

# 현재 파일 경로 기준으로 프로젝트 루트(= module 디렉토리)를 sys.path에 추가
# ex) module/src/prepare_output.py -> module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../module/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) # .../module
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)