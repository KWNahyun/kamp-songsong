# Official GT 일관성·예측 정렬·후보 선택 진단

작성: 2026-10-01. 분석 범위: 새 v2 train/validation만. 공식 TXT 수정 없음. Test 이미지·라벨 사용 없음.

## 결론

반복 촬영 영상의 정렬 후에도 공식 bbox의 위치·크기가 달라지는 사례가 확인됐다. 이는 high-IoU 평가에 라벨 경계 규칙·촬영 변동이 영향을 줄 가능성을 지지한다. 그러나 전역 영상 정렬만으로 실제 이물의 국소 이동·투영 변화까지 제거할 수 없으므로 **라벨 오류나 annotation uncertainty의 비율을 확정하지 않는다.**

이와 독립적으로, D-FINE에서는 공식 GT에 더 정확한 후보가 있지만 최고 점수를 받지 못하는 현상이 명확하다. 따라서 **MAL 재검증은 진행할 가치가 있다.** UQ는 MAL 결과 확인 이후 판단한다. 현재는 좌표 정제 모듈이나 정답 박스 수정을 먼저 적용할 근거가 부족하다.

## 1. GT를 사용하지 않은 영상 정렬

원본 컬러 영상에서 채널 값이 다른 픽셀을 검출하고 3px 팽창해 등록 계산에서 제외했다. 인페인팅한 이미지의 복원 흔적으로 유사도를 판단하지 않도록 원본 회색조와 제품 주변 구조를 사용했다. 공식 GT 좌표는 영상 변환 추정에 사용하지 않았다.

같은 split·장비·날짜·제공 session·파일 prefix·해상도 그룹에서 시간순 인접 영상 중 간격 120초 이하를 비교했다. 이것은 모든 near-duplicate의 전수 검색이 아니라, 반복 촬영 가능성이 높은 인접 쌍의 진단이다. session은 제공 메타데이터이며 동일 물리적 시편 ID가 보장되는 것은 아니다.

- 후보 영상 쌍: 323개
- 양방향 translation ECC로 정렬
- 마커와 영상 가장자리를 제외한 NCC≥0.97
- 정·역방향 이동 합의 오차≤0.5px, 이동 크기≤8px
- 통과: 87쌍, 15개 장비·날짜 그룹
- 정렬 후 GT 중심거리 3px 이하에서 일대일 대응: 153개 bbox 쌍, 13개 장비·날짜 그룹

GT 중심 대응은 정렬 완료 후에만 사용했다. 큰 라벨 중심 차이를 가진 사례는 3px 제한으로 빠질 수 있으므로 선택 편향이 있다. 확대·회전은 정렬하지 않았으며, NCC가 높아도 작은 이물 자체가 동일하다는 보장은 없다.

## 2. 정렬된 GT의 차이

| 지표 | 결과 |
|---|---:|
| GT IoU 중앙값 | 0.650 |
| GT 중심거리 중앙값 | 1.42px |
| 네 경계 차이 중 최댓값의 중앙값 | 2.20px |
| GT IoU<0.75 | 125/153쌍 |
| GT IoU<0.5 | 6/153쌍 |
| 중심거리≤1px인 쌍 | 46쌍 |
| 위 46쌍 중 IoU<0.75 | 25쌍 |

중심 차이가 작은 일부 쌍에서도 폭·높이 차이가 남아, 단순한 영상 전체 이동만으로 bbox 차이를 설명하기 어렵다. 예시에는 유사한 점 주변의 박스 높이가 약 1.8배 달라지는 경우가 있었다. 다만 동일한 점의 독립 재라벨링 실험은 아니므로 이 숫자를 annotation error rate라고 부르지 않는다. 같은 영상이 여러 쌍에 들어가며 독립 표본 153개도 아니다.

![정렬된 반복 촬영 GT](<../../../experiments/kamp_v2_baselines/alignment_audit/aligned_GT_examples.png>)

왼쪽은 영상 A와 공식 박스, 오른쪽은 A에 정렬한 영상 B와 공식 박스다. 원본 마커는 시각화에 남아 있으나 정렬에는 색상 마커 주변을 제외했다. 전역 정렬 후에도 이물 점의 국소 위치가 조금 달라지는 사례가 있어 GT 차이 전체를 라벨 문제로 귀속하지 않는다.

## 3. 4개 모델 설정 × 3 seed의 GT alignment

12개 run/checkpoint를 비교했다. 독립적인 아키텍처 12개가 아니다. 각 GT 주변 IoU≥0.1 후보 중 최고 점수 후보를 기준으로 중심·폭·높이·네 경계 오차를 기록했다. 이 선택은 실제 점수 기준의 후보이며 best-IoU oracle와 구분한다.

| 설정 | 중심 x 오차 중앙값 | 중심 y 오차 중앙값 | 폭 오차 중앙값 | 높이 오차 중앙값 |
|---|---:|---:|---:|---:|
| YOLOv8s | +0.157px | +0.100px | +0.227px | +0.561px |
| YOLOv8s P2 | +0.154px | +0.131px | +0.112px | +0.482px |
| D-FINE-S | +0.105px | −0.071px | +0.178px | +0.539px |
| D-FINE-S P2 | +0.125px | +0.007px | +0.099px | +0.536px |

144개 validation GT 중 48개는 12개 run의 주변 최고 점수 후보가 모두 IoU .75 미만이었다. 99개는 폭 또는 높이 오차가 12개 run 모두 같은 부호로 0.5px를 초과했다. 이는 공통적인 alignment 패턴의 존재를 뜻한다. 모든 모델이 같은 GT·유사한 손실로 학습했으므로 공통 학습 편향일 가능성도 남는다. 각 GT의 네 경계 오차, run 간 표준편차와 부호 일치 수는 CSV로 보관했다.

Signed median이 작다는 사실은 각 사례의 오차가 작다는 뜻이 아니다. 과대·과소 박스가 상쇄될 수 있다. 또한 공통 오차와 특정 설정 오차는 연속적인 현상으로, 임의의 단일 cutoff로 원인을 확정하지 않는다. 모델별 사례 시각화는 앞선 [실패 사례](<../../../experiments/kamp_v2_baselines/analysis/failure_examples.png>) 및 이번 GT 쌍과 함께 검토한다.

![설정·seed별 박스 정렬](<../../../experiments/kamp_v2_baselines/alignment_audit/model_seed_alignment.png>)

첫 행은 공통 실패 GT110, 나머지는 설정별 평균 IoU 차이가 큰 세 GT다. 녹색은 공식 박스, 빨강·주황·청록은 세 seed의 주변 최고 점수 후보다. 이미지별로 같은 crop을 네 설정에 사용했다. 차이가 큰 사례를 의도적으로 뽑았으므로 전체 분포의 대표 표본은 아니다.

## 4. Ranking/selection은 별도로 판단

동일 GT·동일 run에서 best-IoU 후보와 주변 최고 점수 후보를 비교했다. 실제 추론에서는 GT를 사용할 수 없으므로 best-IoU는 진단 상한이다.

| 설정 | IoU≥.75 후보가 있지만 최고 점수 후보는 .75 미만인 GT 평균 | 최선 IoU−최고 점수 후보 IoU 평균 |
|---|---:|---:|
| YOLOv8s | 0.00 | 0.00015 |
| YOLOv8s P2 | 0.00 | 0.00012 |
| D-FINE-S | 49.00 | 0.08772 |
| D-FINE-S P2 | 41.67 | 0.07526 |

YOLO는 이미 NMS 이후 예측이므로, 0이라는 값으로 NMS 전 ranking 문제가 없다고 결론 내리면 안 된다. D-FINE은 NMS 전 query 출력이다. 두 계열의 위 숫자를 직접 우열 비교하는 표가 아니다.

기존 내부 추적에서 NMS로 IoU .75 후보가 전부 사라지는 GT는 D-FINE 평균 36개, P2 32개였다. 중간 단계의 정밀 후보가 최종 decoder에서 사라지는 경우도 각각 15개와 18개였다. 따라서 ranking, NMS 대표 선택, decoder refinement는 각각 관찰되며 하나의 원인으로 합치지 않는다.

공식 라벨의 경계에 불확실성이 있어도, **현재 공식 기준에 더 잘 맞는 후보가 존재하는데 점수 때문에 선택되지 않는 현상**은 별도로 검증할 수 있다. 이것이 MAL/UQ를 검증할 직접적인 이유다. 높은 공식 IoU가 실제 물리적 경계의 정확성을 증명하는 것은 아니다.

## 5. 증거별 판단과 후속 실행

| 가설 | 현재 증거 | 결정 |
|---|---|---|
| 라벨 경계 규칙·촬영 변동 영향 | 정렬된 GT 쌍에서도 크기·경계 차이 | 가능성 지지, 라벨 오류로 확정하지 않음 |
| 모델 공통 localization 편향 | 12개 run의 동일 방향 오차 | 라벨 영향과 공통 학습 편향을 완전히 분리하지 못함 |
| 모델별 localization 차이 | 설정·seed별 오차가 존재 | 공통 오류와 별도 기록, 직접 residual 추가 보류 |
| Ranking 불일치 | D-FINE 평균 49 GT에서 정밀 후보가 최고 점수가 아님 | MAL 재검증 진행 |
| Duplicate/selection | NMS가 낮은 FP Recall을 높이지만 정밀 후보 손실 | 동일 NMS 대조 유지, UQ는 MAL 결과 후 판단 |
| Decoder refinement 불안정 | 중간 정밀 후보 일부 소실 | MAL 이후 잔존 병목이면 검토 |

후속은 새 데이터의 D-FINE-S+MAL 세 seed, 각 30 epoch다. baseline과 같은 COCO 초기값·분할·입력·optimizer 설정을 유지하고 VFL→MAL만 변경한다. 크기 loss·P2·UQ는 추가하지 않는다. Train 실제 batch의 유한 loss/gradient 검증 후 본 학습을 실행하고, 완료 예상 이후 공통 평가와 분석을 예약한다. Scale 1.0과 동일 NMS .7을 주 비교로 사용한다.

실행 기록: train batch 3회 검증 통과, peak VRAM 5,091 MiB. 21:37 KST에 세 seed 순차 학습 시작, 예상 약 20분. 22:00 KST에 완료 확인·공통 평가·실패 분석 예약. 실시간 확인은 `python3 /home/viplab/contest/experiments/kamp_v2_mal/watch.py`.

## 재현 파일

- [영상 정렬 기록](<../../../experiments/kamp_v2_baselines/alignment_audit/registration.json>)
- [정렬 GT 쌍](<../../../experiments/kamp_v2_baselines/alignment_audit/matched_GT_pairs.csv>)
- [예측 정렬](<../../../experiments/kamp_v2_baselines/alignment_audit/prediction_alignment.csv>)
- [run 간 일치도](<../../../experiments/kamp_v2_baselines/alignment_audit/cross_run_agreement.csv>)
- [ranking 비교](<../../../experiments/kamp_v2_baselines/alignment_audit/ranking.csv>)
- [진단 요약](<../../../experiments/kamp_v2_baselines/alignment_audit/summary.json>)
