#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/viplab/contest"
REPORT_PATH="$PROJECT_ROOT/reports/합성데이터_학습_분석보고서.md"
LOG_DIR="$PROJECT_ROOT/experiments/synthetic_scheduled_analysis"
mkdir -p "$LOG_DIR"

CODEX_BIN=$(find /home/viplab/.vscode/extensions /home/viplab/.vscode-server/extensions \
  -maxdepth 5 -type f -path '*/openai.chatgpt-*-linux-x64/bin/linux-x86_64/codex' \
  -printf '%p\n' 2>/dev/null | sort -V | tail -n 1)

if [[ ! -x "$CODEX_BIN" ]]; then
  echo "최신 VS Code Codex 실행 파일을 찾지 못했습니다: $CODEX_BIN" >&2
  exit 1
fi

PROMPT=$(cat <<'EOF'
/home/viplab/contest에서 합성 데이터 관련으로 실행 중이거나 완료된 모든 학습을 분석한다.

1. 프로세스, 로그, 체크포인트, 결과 파일을 확인한다.
2. 완료된 실험은 데이터셋 구성, 실험 목적, 핵심 정량 지표, 조건별 성능, 실패 사례, 임계값에 따른 변화 가능성을 분석한다.
3. 아직 실행 중인 실험은 진행률, 예상 남은 시간, 다음 확인 시점을 기록한다.
4. 기존 실데이터 최종 모델, 임계값, 시험셋 설정을 변경하거나 재튜닝하지 않는다.
5. 가독성 개선본이나 제출용 보고서는 수정하지 않는다.
6. 사용자 요청에 따른 별도 보고서 예외로 /home/viplab/contest/reports/합성데이터_학습_분석보고서.md를 새로 작성하거나 갱신한다.
7. 보고서 맨 위에는 비전공자도 이해할 수 있는 쉬운 요약을 쓴다. 이어서 결과 해석, 제출 보고서에 활용할 수 있는 서술 문장, 넣을 표·그림 제안, 해석 범위와 다음 확인 항목을 작성한다.
8. 수치와 근거 파일 경로를 명시하고, 합성 데이터 결과를 실제 현장 성능 증명으로 과장하지 않는다.
EOF
)

RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
FINAL_MESSAGE="$LOG_DIR/final_$(date +%Y%m%d_%H%M%S).md"

exec "$CODEX_BIN" exec \
  --ephemeral \
  --skip-git-repo-check \
  --model gpt-6.1-sol \
  --config 'model_reasoning_effort="medium"' \
  --sandbox workspace-write \
  -C "$PROJECT_ROOT" \
  --output-last-message "$FINAL_MESSAGE" \
  "$PROMPT" >>"$RUN_LOG" 2>&1
