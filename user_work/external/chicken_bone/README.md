# X-ray 이물 detector ablation용 소형 외부 데이터

구축일: 2026-09-29. **바로 사용할 설정: [dataset/data.yaml](dataset/data.yaml)**

팀의 KAMP 전처리 데이터가 오기 전 구조 비교용으로 구축했다. 식품 X-ray라는 공통점이 있으나 KAMP 성능을 대신하는 벤치마크는 아니다. 모델 학습은 아직 실행하지 않았다.

## 구성

| 분할 | 영상 | 뼈 포함 | 정상 | 박스 | 닭고기 ID | 뼈 ID |
|---|---:|---:|---:|---:|---|---|
| train | 154 | 88 | 66 | 88 | 1–9 | 1–18 |
| val | 40 | 30 | 10 | 30 | 10–11 | 19–28 |
| test | 60 | 45 | 15 | 45 | 12–14 | 29–43 |
| 합계 | 254 | 163 | 91 | 163 | 14개 | 43개 |

원 배포 설명의 전체 촬영 뼛조각 수와 이번 선택 조건의 고유 Bone_ID 수는 다르다. 위 표는 실제 다운로드한 CSV를 기준으로 집계했다. 같은 뼈를 위치만 바꿔 찍은 영상은 같은 분할에 유지했다. 원본 train+val을 새 train으로 합치고, 원본 test를 닭고기 ID 기준으로 새 val/test로 나눴다. **공식 benchmark split을 재현한 구성이 아니라 프로젝트용 분할**이다. 평가 분할은 시료 3개뿐이므로 여러 영상이 독립 표본인 것처럼 신뢰구간을 계산하지 않는다.

## 입력과 라벨

- 입력: 352×280 PNG, 8bit RGB(세 채널 동일한 회색조). 원본 956×760의 종횡비를 유지한 축소이며 패딩은 하지 않았다.
- 원본 조건: Dataset #1 / `40kV_40W_100ms_10avg`만 선택. 다른 전압·노출 조건은 미사용.
- TIFF는 이미 flat/dark-field와 로그 보정된 감쇠 영상이다. 추가 로그 보정을 하지 않는다.
- 학습 영상의 픽셀을 8픽셀 간격으로 샘플링해 pooled 0.1/99.9 백분위로 고정 표시 범위를 추정했다. 모든 분할에 같은 범위를 적용하고 감쇠가 큰 부분이 어두워지도록 반전했다. 수치는 [summary.json](dataset/summary.json)에 기록했다. 영상별 자동 대비, CLAHE, 인페인팅, 합성, 정답 기반 입력 변경은 하지 않았다.
- 라벨: KAMP와 동일한 `0 center_x center_y width height` 정규화 YOLO TXT. 클래스 `0=Defect`, 실제 대상은 뼛조각이다. 정상 영상은 빈 TXT다.
- 원본 mask는 2개 채널이다. 0번은 닭고기, **1번만 뼈 정답**이다. CSV상 영상당 하나의 물리적 뼈를 사용했으므로 1번 마스크의 모든 픽셀을 감싸는 박스 하나를 만든다. 끊어진 마스크 조각을 별도 객체로 늘리지 않는다.
- bbox는 반개방 좌표 `[xmin,ymin,xmax+1,ymax+1]`로 생성한 뒤 정확한 가로/세로 배율로 변환한다. YOLO/COCO 간 오차를 검증했다.
- KAMP bbox 너비·높이 중앙값은 각 10px; 이 세트는 약 14.36×14.00px, 짧은 변 중앙값 약 11.79px다. 길쭉한 뼈도 있어 동일 분포는 아니다.
- 학습 로더에서 다시 640으로 확대하면 위 픽셀 크기도 커진다. 비교 모델끼리 동일 입력 크기·letterbox 설정을 유지한다.

```
dataset/
  data.yaml
  classes.txt
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt
  annotations/{train,val,test}.json
  manifest.json
  summary.json
  validation.json
  image_checksums.json
```

COCO annotation의 category id도 0이다. 사용 프레임워크의 category mapping은 이 파일의 `categories`를 따르도록 설정한다. COCO의 file_name은 각 images/split 폴더에 대한 상대 파일명이다.

## 검증과 사용 범위

254개 영상 전부 디코딩, 채널 일치, 박스 범위, 양성/음성 라벨 일치, YOLO–COCO 좌표 일치를 검사했다. 원본·변환 영상의 픽셀 해시가 모두 고유하며 분할 간 닭고기 ID·뼈 ID 중복이 없다. [검증 결과](dataset/validation.json)

[학습 예시](review/train_contact_sheet.jpg), [검증 예시](review/val_contact_sheet.jpg), [평가 예시](review/test_contact_sheet.jpg)는 검토용으로만 박스를 그렸다. `dataset/images`에는 박스 표시가 없다. 예시를 육안 확인했으나 모든 이물의 가시성이나 원 정답 정확도를 재라벨링한 것은 아니다.

고정 분할에서 baseline/P2/head/회귀 변경을 비교하는 소형 실험용이다. 외부 test는 구조 선정에 반복 사용하지 않는다. 제품 종류, 정상 비율, 이물 재질, 촬영 조건이 KAMP와 달라 여기서 고른 판정 임계값을 KAMP에 그대로 적용하지 않는다. 원본은 양성 1박스, 음성 0박스이므로 KAMP의 다중 이물 검출 평가를 대체하지 못한다.

## 출처·재현

- 원 데이터: Vladyslav Andriiashen, *Dual-energy X-ray inspection of chicken fillets containing rib bone fragments* (2024), [Zenodo DOI](https://doi.org/10.5281/zenodo.10579608).
- 라이선스: CC BY 4.0, [기관 확인](https://ir.cwi.nl/pub/33988). 가공 내용과 저자 출처를 함께 명시한다.
- [저자 코드](https://github.com/vandriiashen/pod2settings)는 마스크 채널의 의미 확인에 참고했다. 데이터 라이선스와 코드 라이선스를 혼동하지 않는다.
- [원본별 다운로드 기록](source/download_manifest.json): ZIP HTTP 범위 요청으로 선택 조건만 추출. 각 멤버의 ZIP CRC32와 크기를 검증하고 SHA256을 기록했다. 전체 ZIP MD5는 전체 파일을 받지 않았으므로 검증하지 않았다.
- `source/Submission`에 선택 조건의 원본 TIFF/마스크/CSV를 보존했다. 학습에는 `dataset`만 필요하다(약 18MB).

재구축 순서:

```bash
python scripts/inspect_source.py
python scripts/download_subset.py
python scripts/build_dataset.py
python scripts/validate_dataset.py
```

[requirements.txt](requirements.txt)에 변환 환경을 기록했다. 모델 학습 라이브러리는 포함하지 않는다. 폴더를 옮기면 `dataset/data.yaml`의 path를 새 dataset 경로로 바꾼다.
