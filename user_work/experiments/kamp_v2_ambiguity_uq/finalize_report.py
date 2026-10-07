from pathlib import Path
import json
p=Path('/home/viplab/contest/프로젝트_진행과정_통합보고서.md');s=p.read_text();a=s.index('### 13.9 ');b=s.index('## 14.',a)
s=s[:a]+'''### 13.9 마지막 제한 UQ 비교 — 9run 완료 및 판단

2026-10-02 22:33:05~22:42:57(KST),9run 학습·평가를 약9분52초에 완료했다. 실패0이다. 이번 ambiguity 묶음의 MAL/edge/soft1px 각각3seed에 기존과 동일한 unary UQ를 적용했다. Detector의 `best_stg1.pth`는 동결하고 UQ62,007개 parameter만 전체 train350장으로30epoch 학습했다. 기존 UQ와 동일 초기화·optimizer·학습률·quality target·score 결합을 사용하고 마지막 epoch를 평가했다. UQ checkpoint를 val 최고점으로 고르지 않았다. Val66장/144GT, NMS .7, bbox scale1, 동일 원본 좌표 평가이며 test는 추가 사용하지 않았다.

완료 후9개 checkpoint의 detector tensor를 각각 원래 checkpoint와 직접 비교해 전부 bitwise 동일함을 확인했다. 원래 detector와 최종 UQ checkpoint의 기록된 해시도 모두 일치했다. 모든 GT의 raw best-IoU 값도 UQ 전후 동일하다. 즉 UQ는 bbox를 새로 정제한 실험이 아니라 score를 바꾼 실험이다.

#### 13.9.1 같은 묶음의 결과 비교

표는3seed 평균±표본표준편차다. TP는 각 run의 validation-selected FP≤6 운영점에서의 검출 수/144GT다. 배포용 고정 임계값 성능이 아니다.

| 모델 | AP | AP75 | TP (FP≤6) | 정밀 raw 후보 보유 GT | 정밀 최고점수 근처 후보 GT | NMS 후 정밀 후보 보유 GT |
|---|---:|---:|---:|---:|---:|---:|
| MAL+UQ | 37.60±0.27 | 15.80±1.49 | 140.33 | 101.67 | 49.00 | 65.00 |
| edge+UQ | 38.62±1.11 | 17.95±2.15 | 141.00 | 98.67 | 51.67 | 70.00 |
| Soft1+UQ | 38.93±0.30 | 18.54±2.14 | 141.00 | 106.00 | 51.67 | 77.33 |

정밀 후보는 official GT IoU≥.75다. 최고점수 근처 후보는 GT IoU≥.3인 후보 중 최고 score를 뜻한다. 이 지표들은 GT를 사용하는 oracle 진단이며 실제 추론 규칙 또는 일대일 recall이 아니다.

| Soft1+UQ의 paired 차이 | 평균 ΔAP | AP 개선 seed | 평균 ΔAP75 | AP75 개선 seed | 평균 ΔTP |
|---|---:|---:|---:|---:|---:|
| 이번 MAL+UQ 대비 | +1.33 | 3/3 | +2.73 | 3/3 | +0.67 |
| 이번 edge+UQ 대비 | +0.31 | 2/3 | +0.59 | 1/3 | 0.00 |

Edge+UQ 대비 seed순(20260929/30/20261001) AP75 차이는 **−0.60 / −0.57 / +2.93**다. 따라서 평균 AP75 이득은 마지막 seed가 끌어올린 결과이며, 일관된 ambiguity-specific 개선으로 볼 수 없다. Edge parameterization 자체의 효과를 통제하지 않고 MAL 대비 수치만 제시하면 해석이 과장된다. 과거 MAL+UQ 수치는 이번 paired 비교에 섞지 않았다.

실행 전 기준은 두 대조군 각각보다 평균 AP/AP75가 높고 각 지표에서2/3seed 이상 개선되며 평균 TP가 감소하지 않는 것이었다. **MAL 대조군 기준은 통과했지만 edge 대조군의 AP75 기준은 실패했다.** 결과 확인 후 기준을 완화하지 않는다. 이는 탐색 종료를 위한 사전 기준이지 통계적 유의성 검정은 아니다.

#### 13.9.2 UQ가 실제 바꾼 것과 바꾸지 못한 것

| 동일 detector에 UQ 추가 | 평균 ΔAP | 평균 ΔAP75 | 평균 ΔTP | 평균 Δ정밀 최고점수 후보 | 평균 ΔNMS 후 정밀 후보 |
|---|---:|---:|---:|---:|---:|
| MAL | +0.40 | +0.37 | 0.00 | −0.33 | −0.33 |
| edge | +1.19 | +1.78 | 0.00 | 0.00 | +1.33 |
| Soft1 | +1.26 | +1.92 | 0.00 | 0.00 | −1.33 |

9run 모두 UQ로 AP/AP75가 올랐지만 **FP≤6 TP는 각 run에서 모두 그대로**다. Soft1의 정밀 최고점수 근처 후보 수는3seed 모두 변화가 없고 NMS 후 정밀 후보 보유 GT는각각1/1/2개 줄었다. Soft1+UQ의 NMS 이후 후보 보존이 다른 모델보다 높은 것은 주로 Soft1 detector에 이미 존재한 이득이다. UQ가 그 후보를 더 잘 보존했다고 말할 수 없다.

UQ는 score 변환이므로 AP 향상은 score에 따른 전체 prediction 순위와 일부 NMS 선택 변화로 발생할 수 있다. 이번 AP 향상을 “같은 이물의 좋은 박스와 더 좋은 박스 구별을 해결했다”거나 “미탐을 줄였다”로 바꾸어 말하지 않는다. 정밀 후보 수가 같아도 후보의 정체가 일부 바뀔 가능성까지 배제한 것은 아니며, 개수 지표와 AP는 다른 질문에 답한다.

#### 13.9.3 남은 실패와 조건별 결과

GT110·136은9run 모두 FP≤6에서 미탐이다. GT110 raw best IoU는 .406~.730, 최고점수 근처 후보는 .364~.454다. GT136 raw best IoU는 .509~.840, 최고점수 근처 후보는 .371~.442다. 특히 Soft1 seed20260930은 GT136에 IoU .840 후보가 있지만 최고점수 근처 후보는 .417이다. 좋은 후보의 존재와 최종 대표 선택 문제를 여전히 구분해야 한다.

| 모델 | M1 TP/56 | M2 TP/42 | M3 TP/46 | 최고점수 후보 경계 MAE(px) |
|---|---:|---:|---:|---:|
| MAL+UQ | 54.67 | 42.00 | 43.67 | .994 |
| edge+UQ | 55.33 | 41.67 | 44.00 | .981 |
| Soft1+UQ | 55.67 | 42.00 | 43.33 | .985 |

Soft1의 M1 이득을 M3 개선으로 일반화할 수 없다. 정밀 raw 후보가 있는데 최고점수 후보는 정밀하지 않은 GT는 MAL52.67, edge47.00, Soft1 54.33개다. Soft1이 후보 풀을 개선한 제한적 신호와, 실제 선택 병목이 남았다는 증거가 함께 존재한다.

#### 13.9.4 최종 결정과 다음 작업의 범위

**이번 ambiguity 모델 탐색을 종료하고 기존 MAL+UQ 비교 기준을 유지한다.** Soft1+UQ는 평균 지표가 유망한 실험 후보로 보존하되, edge 대조군 대비 일관성 기준을 통과하지 못했으므로 새로운 최종 모델로 교체하지 않는다. 새로운 loss 조합·큰 모듈·추가 seed·test 재평가를 시작하지 않았다.

Annotation ambiguity의 시각적 관찰이 틀렸다는 뜻은 아니다. 관찰을 활용한 감독 완화가 강한 대조군을 안정적으로 넘는다는 증거가 부족하다는 결론이다. 동일 영상의 독립 반복 annotation은 아직 없으므로 실제 annotator variance나 최적 ε를 측정했다고 주장하지 않는다. Official TXT는 유지한다.

이후에는 새 모델 탐색보다 (1) 검은 점과 official box의 경계 규칙을 사례로 명시하고, (2) 객체 IoU 미탐과 제품 단위 경보 실패를 분리하며, (3) 장비·대비·후보 불일치별 재검사 우선순위를 기존 출력으로 정리하는 작업이 적절하다. 앞선 재검사 신호의 안정적 이득 미확인도 그대로 보고한다. 정상 제품과 독립 촬영 데이터 없이 안전한 자동 통과 기준이나 실제 재검사 회수율을 확정하지 않는다. 사람의 추가 검사 없이 “놓친 이물을 회수했다”고 계산하지 않는다. 향후 독립 촬영 그룹을 얻으면 고정 기준과 보존 후보를 검증할 수 있다.

사용자 요청에 따라 완료 확인 예약 `ablation`은 **삭제**했다. 추가 자동 학습/분석 예약은 이 실험에 남기지 않았다.

[고정 계획](experiments/kamp_v2_ambiguity_uq/manifest.json) · [run별 결과](experiments/kamp_v2_ambiguity_uq/analysis/per_run.csv) · [paired 비교](experiments/kamp_v2_ambiguity_uq/analysis/paired_UQ.csv) · [UQ 추가 효과](experiments/kamp_v2_ambiguity_uq/analysis/UQ_gain.csv) · [반복 미탐](experiments/kamp_v2_ambiguity_uq/analysis/repeated_misses.csv) · [검증 결과](experiments/kamp_v2_ambiguity_uq/analysis/verification.json)

'''+s[b:]
a=s.index('**최신 완료 결과');b=s.index('\n\n',a)
s=s[:a]+'''**최신 완료 결과(2026-10-02):** ambiguity36회에 이어 MAL/edge/Soft1 각각3seed의 동일 UQ 비교9회를 완료했다. Soft1+UQ는 이번 MAL+UQ 대비 AP+1.33/AP75+2.73이지만 edge+UQ 대비 AP+0.31/AP75+0.59이며 AP75 개선은1/3seed뿐이다. 사전 일관성 기준을 충족하지 못했다. UQ로 AP는 개선됐지만 FP≤6 TP는 각 run에서 그대로이고 GT110·136도 남았다. 기존 MAL+UQ 기준을 유지하고 이번 모델 탐색을 종료한다. 상세 결과는 §13.9다. 예약은 사용자 요청으로 삭제했다.'''+s[b:]
s=s.replace('MAL/edge/Soft1 + 동일 UQ 9run 진행 중; 기존 모델 교체 안 함','동일 UQ 9run도 완료, 실패0. 일관성 기준 미충족으로 탐색 종료; 기존 모델 유지')
a=s.index('**36회 분석은 완료했고, 사용자 승인으로');b=s.index(' 양성 영상의',a)
s=s[:a]+'''**36회 ambiguity 실험과9회 UQ 후속 비교·분석을 모두 완료했다.** Soft1+UQ의 평균 이득은 있지만 edge 대조군 대비 AP75 일관성 기준을 충족하지 못해 모델 탐색을 종료하고 기존 기준을 유지한다. 재현 감사는 §13.8, 완료 결과와 판단은 §13.9를 따른다. 독립 반복 annotation에 의한 경계 variance 측정은 아직 미실행이다. 예약은 삭제했다.'''+s[b:]
p.write_text(s)
e=Path(__file__).parent/'analysis/completion.json';d=json.loads(e.read_text());d.update(report_update_pending=False,analysis_complete=True,automation_deleted=True,decision='retain existing baseline; stop ambiguity model search');e.write_text(json.dumps(d,indent=2))
