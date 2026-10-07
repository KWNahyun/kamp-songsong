#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command="${1:-help}"
[[ $# -gt 0 ]] && shift

case "$command" in
  help|-h|--help)
    cat <<'USAGE'
KAMP unified runner

  ./run.sh final <command> ...      Final MAL+UQ package; see ./repro/run.sh help
  ./run.sh data-build               Build marker-removed dataset (requires raw inputs + EDA outputs)
  ./run.sh handoff-build            Build KAMP train/val/test handoff package
  ./run.sh synth-test               Generate gVXR/replica stress-test set

Before data/synthesis commands, set KAMP_ROOT to the repository root and
prepare the raw data plus intermediate EDA/calibration assets described in
docs/DATA_AND_SYNTHESIS.md.
USAGE
    ;;
  final)
    exec "$ROOT/repro/run.sh" "$@"
    ;;
  data-build)
    export KAMP_ROOT="${KAMP_ROOT:-$ROOT}"
    exec python3 "$ROOT/prep/build_dataset.py"
    ;;
  handoff-build)
    export KAMP_ROOT="${KAMP_ROOT:-$ROOT}"
    exec python3 "$ROOT/prep/build_handoff.py"
    ;;
  synth-test)
    export KAMP_ROOT="${KAMP_ROOT:-$ROOT}"
    exec python3 "$ROOT/prep/synth/build_synth_test.py"
    ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
