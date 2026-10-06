#!/bin/bash
# 새 모델 변형(사전학습 ①·③, 옮겨 심기 학습)이 seed 3개 모두 끝나면 평가를 순서대로 자동 실행 (추론만, 한 번에 하나 → 결과 파일 경합 없음)
# 평가: 기본 평가(nb2_eval) → 실패 조건 → 합성 검증 V3·V4(지운 자리 반응) → 위치 축 스트레스(기증 train / val·test)
# 사용법: run_variant_eval.sh <gpu> <변형 ...>   예) run_variant_eval.sh 1 yolov8s_pidray dfine_s_scratch yolov8s_tpB
K=/data/knhyun/KAMP; PY=$K/.venv/bin/python; G=$1; shift; cd $K
F="meshgrid|_VF|Model Summary|HGNetV2|network|Downloading|%\|"
todo="$*"
while [ -n "$todo" ]; do
  left=""
  for v in $todo; do
    if grep -qh "DONE ${v}_s2" runs/nb2/queue_*.log 2>/dev/null; then
      L=$K/eval/out_nb2/eval_$v.log
      echo "== $v 시작 $(date +%T)"
      ( export NB2_MODEL=$v CUDA_VISIBLE_DEVICES=$G
        echo "== 1 기본 평가 $(date +%T)"; $PY eval/nb2_eval.py $v 2>&1 | grep -vE "$F"
        echo "== 2 실패 조건 $(date +%T)"; $PY eval/failure/collect_heldout.py 2>&1 | grep -vE "$F" | tail -3; $PY eval/failure/failure_analysis.py 2>&1 | grep -vE "$F" | tail -3
        echo "== 3 합성 검증 V3·V4 $(date +%T)"; $PY eval/synth/validate_replica.py --infer 2>&1 | grep -vE "$F" | tail -12
        for dn in train heldout; do echo "== 4 위치 축 $dn $(date +%T)"; STRESS_DONOR=$dn $PY eval/synth/stress_pos.py 2>&1 | grep -vE "$F" | tail -22; done
        echo "== DONE $v $(date +%T)" ) > $L 2>&1
      echo "== $v 끝 $(date +%T)"
    else
      left="$left $v"
    fi
  done
  todo=$(echo $left)
  [ -n "$todo" ] && sleep 120
done
echo "== ALL DONE $(date +%T)"
