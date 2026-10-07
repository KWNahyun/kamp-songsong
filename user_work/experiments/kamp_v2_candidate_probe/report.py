from pathlib import Path
import csv,json
import numpy as np
E=Path(__file__).parent;R=E.parents[1]
rs=list(csv.DictReader((E/'summary.csv').open()))
lines=['# 새 데이터 v2 후보 상대 품질 구별 가능성 진단','', '작성: 2026-10-02. 기본 D-FINE/MAL × 3 seed의 동결 checkpoint. Train350장·797 GT로 진단용 선형 모델을 학습하고 validation66장·144 GT에서 확인했다. Test 및 공식 TXT 변경 없음.','', '## 질문과 방법','', 'UQ는 전역 AP를 개선했지만 동일 이물 주변의 대표 후보는 거의 바꾸지 않았다. 이번 질문은 기존 query 특징과 경계 분포에 공식 GT 기준으로 더 정확한 후보를 구별할 정보가 있는가다.','', '최종 300개 query에서 score≥.001, 유효 크기, GT IoU≥.1을 만족하는 후보를 해당 후보의 최대 IoU GT에 독점 할당했다. GT를 이용한 후보 묶음은 추론에서 사용할 수 없는 oracle 조건이다. 따라서 아래 값은 탐지 AP나 실제 사용 가능한 재검사 성능이 아니다.','', '각 GT 내 특징·IoU 평균을 빼 상대 차이를 학습하는 선형 ridge probe를 사용했다. 각 GT의 총 학습 가중치를 같게 두고 train에서만 스케일을 계산했다. Ridge 계수 .1을 고정했으며 validation으로 튜닝하지 않았다. Score, geometry+score, FDR분포통계+geometry+score, query feature, 전체 특징의 다섯 구성을 비교했다. 쌍 정확도는 IoU 차이≥.05인 후보 쌍의 우열을 맞힌 비율을 GT별 평균했다.','', '## 결과','', '모든 수치는 세 seed 평균. 정밀 선택 수는 대표 후보 IoU≥.75인 GT 수이며, 최대 144개다. 동일 후보 집합의 base-score 대표와 비교한다.','', '| 부모 | 입력 | train 쌍 정확도 | val 쌍 정확도 | val 대표 평균 IoU | 기존→probe 정밀 선택 | 개선/악화 GT |','|---|---|---:|---:|---:|---:|---:|']
for base in ['base','mal']:
 for feat in ['score','geometry_score','distribution_geometry','query_only','all']:
  sub=[r for r in rs if r['base']==base and r['features']==feat and r['split']=='val'];tr=[r for r in rs if r['base']==base and r['features']==feat and r['split']=='train']
  mean=lambda key:np.mean([float(r[key]) for r in sub])
  lines.append(f"| {base} | {feat} | {np.mean([float(r['pair_accuracy']) for r in tr]):.3f} | {mean('pair_accuracy'):.3f} | {mean('probe_mean_iou'):.3f} | {mean('base75'):.1f}→{mean('probe75'):.1f} | {mean('improved'):.1f}/{mean('worsened'):.1f} |")
lines+=['','## seed별 전체 특징 결과','','| 부모 | seed | 기존→probe 평균 IoU | 기존→probe 정밀 선택 | oracle 정밀 coverage |','|---|---|---:|---:|---:|']
for r in rs:
 if r['split']=='val' and r['features']=='all':lines.append(f"| {r['base']} | {r['seed']} | {float(r['base_mean_iou']):.3f}→{float(r['probe_mean_iou']):.3f} | {r['base75']}→{r['probe75']} | {r['oracle75']} |")
lines+=['','## 해석상의 한계','','- 후보 쌍 분류와 최고 후보 선택은 다른 문제다. 쉬운 쌍을 많이 맞혀도 대표 선택이 악화될 수 있다.','- GT로 후보를 묶은 조건이므로 실제 clustering, 배경 억제, NMS와 가까운 복수 이물 간 충돌은 해결하지 않았다.','- 검출기는 해당 train 영상에서 이미 학습됐다. Train 진단 수치는 in-sample이며 일반화 근거는 validation 결과다. Validation은 기존 checkpoint 선택과 반복 탐색에 사용됐으므로 새 독립 검증이 아니다.','- 이번 선형 probe 실패는 모든 비선형/관계 모듈의 불가능성을 뜻하지 않는다. 성공 또한 새로운 아키텍처 성능을 보장하지 않는다.','- 공식 GT와의 정렬 가능성을 측정하며 실제 물리적 이물 경계 정확성은 검증하지 않는다. Seed 세 개와 같은 영상 내 후보는 독립 표본 수천 개로 취급하지 않는다.','', '## 재현 자료','','- [요약](experiments/kamp_v2_candidate_probe/summary.csv)','- [GT별 결과](experiments/kamp_v2_candidate_probe/instances.csv)','- [기존 출력 일치 검증](experiments/kamp_v2_candidate_probe/verification.json)','- [실행 로그](experiments/kamp_v2_candidate_probe/run.log)']
(R/'새_데이터_v2_후보_상대품질_진단.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[10:]))
hard=list(csv.DictReader((E/'hard_pairs.csv').open()))
extra=['## 어려운 후보 쌍과 판단','','전체 특징 probe의 validation 정확도. 두 후보 모두 충분히 GT에 가까운 조건을 별도로 분석했다. 모든 쌍에서 IoU 차이는 .05 이상이다. 조건마다 포함 GT가 달라 단순 난이도 인과효과로 해석하지 않는다.','','| 부모 | 전체 쌍 | 둘 다 IoU≥.5 | 둘 다 IoU≥.65 |','|---|---:|---:|---:|']
for base in ['base','mal']:
 vals=[np.mean([float(r['accuracy']) for r in hard if f'_{base}_' in r['run'] and r['subset']==s])*100 for s in ['all','both_iou_05','both_iou_065']]
 extra.append(f'| {base} | {vals[0]:.1f}% | {vals[1]:.1f}% | {vals[2]:.1f}% |')
extra+=['','**이번 방식에서는 대략적인 품질 차이는 구별했지만, 이미 괜찮은 두 박스 중 더 정확한 박스를 고르는 능력은 약했다.** 둘 다 IoU≥.65인 후보 쌍에서는 평균 약 51%로, 우열을 무작위로 고르는 50% 기준에 가깝다. 유의성 검정을 한 결론은 아니다. MAL의 전체 특징 probe는 정밀 대표 수를 53.3→49.3개로 낮췄고, 세 seed 중 두 개가 악화·한 개가 동일했다. 따라서 강한 relation 모듈을 바로 채택할 근거는 얻지 못했다.','', '## 다음 실험 결정','','1. 잠정 기준 MAL+UQ+NMS .7+scale1.0을 유지한다. 이번 probe를 추론 파이프라인에 넣지 않는다.','2. 다음 작은 실험은 **좋은 후보끼리의 우열에 집중한 학습 목표**를 검증하는 것이다. 기존 전체 후보 목적과 hard-pair/listwise 목적을 같은 고정 특징·같은 용량에서 비교한다. GT는 train의 감독·사후 평가에만 사용하고 실제 후보 묶음은 predicted box overlap으로 만든다.','3. 성공 기준은 쌍 정확도 상승만이 아니다. 동일 후보 기준 대표 IoU/정밀 선택이 개선되고, 실제 NMS 후 AP75가 좋아지며, FP≤6 TP를 잃지 않아야 한다. 기본과 MAL 부모 및 seed별 개선·악화를 모두 기록한다. 검증 결과에 따라 임계값을 반복 조절하지 않는다.','4. 이 통제 실험에서도 정밀 대표 선택이 나아지지 않으면 query 관계망 확대를 보류한다. 그때 로컬 영상/고해상도 특징이 정밀 경계 판단에 추가 정보를 주는지 별도로 진단한다. 이는 P2 전체 경로 재추가나 기존 BR/BPS의 단순 반복과 구분해야 한다.','5. M3 반복 미탐과 라벨 경계 모호성은 별도 축으로 유지한다. 이번 선택 진단만으로 안전한 통과·재검사 기준을 확정할 수 없다.','', '검출기 신규 학습은 실행하지 않았다. 6개 고정 checkpoint에서 특징을 추출하고 총 30개 선형 probe를 학습·평가했다. 396장 validation 재추출에서 기존 박스 최대 차이는 모두 0이었다.','', '[어려운 후보 쌍 상세](experiments/kamp_v2_candidate_probe/hard_pairs.csv)']
p=R/'새_데이터_v2_후보_상대품질_진단.md';s=p.read_text();s=s.replace('## 질문과 방법','**결론: 현재 특징의 단순 상대 점수화는 정밀 후보 선택을 개선하지 못했다. 좋은 후보끼리의 구분과 학습 목표를 먼저 검증하고, 관계 모듈 확대는 보류한다.**\n\n## 질문과 방법');p.write_text(s+'\n'+'\n'.join(extra)+'\n')
(E/'completion.json').write_text(json.dumps(dict(completed=True,runs=6,probes=30,test_used=False,detector_training=False),indent=2))
