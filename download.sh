#!/bin/bash

set -e

FILE_ID="1wRSc_nkGd1TSPm_jQZyx_9hSsZiy6iAL"
ZIP_NAME="4. X-ray 검사장비 AI 데이터셋.zip"
OUTPUT_DIR="xray_dataset"

echo "===== 1. Google Drive 다운로드 ====="

gdown "https://drive.google.com/uc?id=${FILE_ID}" \
    -O "${ZIP_NAME}"

echo
echo "===== 2. 파일 확인 ====="

ls -lh "${ZIP_NAME}"
file "${ZIP_NAME}"

echo
echo "===== 3. ZIP 테스트 ====="

if ! unzip -t "${ZIP_NAME}" > /dev/null; then
    echo "ERROR: 다운로드된 파일이 정상적인 ZIP 파일이 아닙니다."
    echo "파일 크기와 다운로드 상태를 확인하세요."
    exit 1
fi

echo "ZIP 파일 정상 확인!"

echo
echo "===== 4. 압축 해제 ====="

mkdir -p "${OUTPUT_DIR}"

unzip -q "${ZIP_NAME}" -d "${OUTPUT_DIR}"

echo
echo "===== 5. 완료 ====="

echo "Dataset location:"
echo "$(pwd)/${OUTPUT_DIR}"

echo
echo "파일 목록:"
find "${OUTPUT_DIR}" -maxdepth 2 -type f | head -30