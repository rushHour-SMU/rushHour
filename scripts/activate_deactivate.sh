#!/bin/bash

# 권한 chmod +x activate_deactivate.sh

# 가상환경 디렉토리로 이동 (가상환경 디렉토리 경로를 정확히 지정하세요)
cd ~/rushHour  # 예시로 ~/rushHour 디렉토리를 사용했습니다.

# 가상환경 비활성화 (만약 활성화되어 있으면 비활성화)
deactivate

# 잠시 대기 (비활성화 후 몇 초 기다림)
sleep 1

# 가상환경 재활성화
source venv/bin/activate

# 가상환경이 활성화되었는지 확인 (옵션)
echo "가상환경이 활성화되었습니다: $(which python)"

# 끄는건 deactivate 입력