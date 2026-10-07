from pathlib import Path
p=Path('/home/viplab/contest/프로젝트_진행과정_통합보고서.md');s=p.read_text()
s=s.replace('이 후속 학습은 자동 시작하지 않았다.','36회 분석 직후에는 후속 학습을 시작하지 않았으며, 이후 사용자 승인으로 §13.9의 제한된 UQ 비교를 시작했다.')
s=s.replace('완료 분석 후 예약 `ablation`을 PAUSED로 변경했다. 새로운 학습은 시작하지 않았다.','36회 완료 분석 직후 예약 `ablation`을 PAUSED로 변경했으며, 이후 §13.9의 승인된 UQ 비교를 위해 다시 활성화했다.')
insert='''### 13.8 MAL 재현 차이 감사

과거 MAL과 이번 MAL의 점수 차이를 확인하기 위해 실제 기존 실행 경로를 같은 seed20260929로 두 번(old_a/old_b), 이번 실행 경로를 한 번(new_a) 실행했다. 각8 optimizer step만 진단했으며 이 가중치를 새 평가 모델로 사용하지 않는다. 초기 모델 전체 해시·입력 영상·GT·CPU/GPU RNG가 같고 첫 forward 출력과 loss도 정확히 같았다. 그러나 기존 경로끼리도 첫 backward gradient와 첫 optimizer update 후 가중치가 달랐다. 이번 경로와 비교해도 같은 양상이었다.

엄격한 결정성 모드로 실제 D-FINE backward를 실행하자 `grid_sampler_2d_backward_cuda`에 결정적 구현이 없다는 오류가 발생했다. 따라서 seed 고정만으로 기존 학습이 bitwise 재현되지 않는다는 직접 증거를 확보했다. **과거 AP 차이 전체가 이 연산 하나 때문이라고 확정한 것은 아니다.** 전체 학습 경로의 오차 증폭과 다른 환경 요인의 기여는 분리 측정하지 않았다. 이번 묶음 내부의 seed별 대조군 비교를 유지하고 과거 MAL 수치를 직접 섞지 않는다.

[재현 감사 결과](experiments/kamp_v2_repro_audit/summary.json) · [엄격한 결정성 검사](experiments/kamp_v2_repro_audit/strict_check.json)

### 13.9 마지막 제한 UQ 비교 — 학습 진행 중

사용자 승인에 따라 **이번 ambiguity 묶음의 MAL / edge / soft1px 각각3seed, 총9run**에 기존과 동일한 unary UQ를 적용한다. 기존 MAL+UQ는 보존한다. 각 detector의 `best_stg1.pth`를 동결하고 UQ만 전체 train350장으로30epoch 학습한다. 기존 UQ와 같은 초기화·optimizer·학습률·quality target·score 결합을 사용하며 마지막 epoch를 평가한다. UQ checkpoint를 val 최고점으로 고르지 않는다. 비교는 val66장, NMS .7, bbox scale1, 동일 평가 코드로 수행한다. Test는 새로 사용하지 않는다.

사전2batch 검사에서 loss/gradient 유한성과 detector 전체 가중치 불변을 통과했다. 실제3seed 병렬 실행에서 첫5epoch가 정상 진행 중이며 GPU 메모리는 약4.6GB였다. UQ는 detector를 동결하므로 이전 전체 detector 학습보다 가볍다. 현재 속도의 전체9run 학습·평가 예상은 약10~15분이며 실제 완료 시간은 로그로 확인한다. 기존 `ablation` 예약을 재사용해 매시간 완료 여부를 확인하고 완료 시 분석하도록 설정했다. 변화 없는 정상 진행은 알리지 않는다.

**실행 전에 고정한 판단 기준:** Soft1+UQ가 이번 MAL+UQ 및 edge+UQ 각각보다 평균 AP·AP75가 높고, 각 지표에서3seed 중 적어도2seed가 개선되며, validation-selected FP≤6의 평균 TP가 낮아지지 않는지 확인한다. 이 기준도3seed의 탐색적 의사결정 기준이며 통계적 유의성이나 독립 일반화를 뜻하지 않는다. 정밀 후보 보존·최고점수 후보 선택·GT110/136·장비별 실패를 함께 보고한다. 기준을 충족하지 못하면 추가 loss/모듈 탐색을 종료하고 경계 표기 규칙과 재검사 조건, 보고서 정리로 전환한다.

GT 경계 비일관성의 기존 시각적 관찰은 유지한다. 다만 동일 영상에 대한 독립 반복 annotation은 아직 없으므로 관찰을 실제 annotator variance 측정으로 표현하지 않는다. 반복 영상의 실제 위치 변화·정합 오차와 annotation 편차도 혼동하지 않는다. 이번 UQ 비교를 위해 TXT를 수정하거나 검은 점 threshold를 새 정답으로 만들지 않는다.

[고정 비교 계획](experiments/kamp_v2_ambiguity_uq/manifest.json) · [실시간 학습 로그 폴더](experiments/kamp_v2_ambiguity_uq/logs) · [사전 검사](experiments/kamp_v2_ambiguity_uq/runs_frozen/dfine_mal_UQ_seed20260929_smoke/metadata.json)

'''
s=s.replace('## 14. 현재 결정·미실행 사항',insert+'## 14. 현재 결정·미실행 사항')
s=s.replace('| GT 경계 uncertainty | 36회 학습·평가·분석 완료, 실패0. Soft1만 검증 후보 보존; 기존 모델 교체 안 함 |','| GT 경계 uncertainty | 36회 분석 완료, 실패0. MAL/edge/Soft1 + 동일 UQ 9run 진행 중; 기존 모델 교체 안 함 |')
old='**학습·평가·결과 분석은 모두 완료했고 추가 학습은 실행하지 않았다.** Soft1px에 제한적 개선 신호가 있지만 기존 MAL+UQ를 교체할 근거는 부족하다. 같은 영상의 반복 annotation과, 필요 시 MAL/edge/soft1에 동일 UQ를 적용하는 작은 후속 비교가 다음 후보이며 아직 미실행이다.'
new='**36회 분석은 완료했고, 사용자 승인으로 MAL/edge/soft1에 동일 UQ를 적용하는9회 제한 비교를 진행 중이다.** Soft1px에 제한적 개선 신호가 있지만 기존 MAL+UQ를 교체할 근거는 아직 부족하다. 재현 감사는 §13.8, 현재 비교와 종료 기준은 §13.9를 따른다. 독립 반복 annotation에 의한 경계 variance 측정은 아직 미실행이다.'
assert old in s;s=s.replace(old,new);p.write_text(s)
