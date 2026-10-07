# 테스트 예측결과 생성 완료

GitHub에 추가된 대표 가중치의 SHA-256을 확인하고 마커 제거 테스트 84장 전체를 실제 추론했습니다.

- `test_predictions.csv`: 원본 영상 좌표의 검출 후보를 표로 확인
- `test_predictions.json`: 동일 후보의 JSON
- `test_alarm_predictions.csv`: 고정 경보 임계값 이상 후보 213개만 보기
- `image_summary.csv`: 테스트 84장의 이미지별 검출 수를 표로 보기
- `image_summary.json`: 검출이 없는 경우를 포함한 모든 84장 처리 기록
- `overlays/`: 고정 검증 임계값 이상 박스를 표시한 영상
- `test_metrics.json`: 공식 COCO 라벨에 대한 실제 평가 결과
- `run_metadata.json`: 모델 해시, 환경, 입력 처리, 실행 시간

이번 CPU 실행의 테스트 AP는 38.55465015227687, AP50은 94.45480972250748입니다. 기록된 원래 validation AP와 이번 validation AP는 약 0.55포인트 차이가 있어 원래 수치와의 완전 일치는 확인되지 않았습니다. 자세한 내용은 루트 검수 보고서를 확인하세요.
