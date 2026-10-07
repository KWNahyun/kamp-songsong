#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'USAGE'
KAMP final-model runner

Usage:
  ./repro/run.sh setup [VENV_PATH]
  ./repro/run.sh predict --images PATH --output PATH
  ./repro/run.sh evaluate --predictions PATH --annotations PATH --output PATH
  ./repro/run.sh prepare-data --dataset PATH --output PATH
  ./repro/run.sh configure-train --data640 PATH --output PATH
  ./repro/run.sh train-mal --config PATH --init PATH --seed SEED [extra D-FINE args]
  ./repro/run.sh train-uq --config PATH --base-checkpoint PATH --output PATH --seed SEED

Dataset paths must refer to the marker-removed KAMP package. `predict` never
reads TXT labels. `evaluate` consumes labels only after predictions are saved.
The `setup` command creates a virtual environment; install a CUDA-compatible PyTorch build, then the project dependencies before GPU inference.
USAGE
}

command="${1:-help}"
[[ $# -gt 0 ]] && shift
case "$command" in
  help|-h|--help) usage ;;
  setup)
    venv="${1:-$ROOT/.venv}"
    "$PYTHON" -m venv "$venv"
    "$venv/bin/python" -m pip install --upgrade pip
    echo "Install CUDA-compatible torch/torchvision first, then run:"
    echo "  $venv/bin/python -m pip install -r $ROOT/requirements.txt"
    ;;
  predict)
    "$PYTHON" "$ROOT/predict.py" --preprocessed "$@"
    ;;
  evaluate)
    "$PYTHON" "$ROOT/evaluate.py" "$@"
    ;;
  prepare-data)
    "$PYTHON" "$ROOT/scripts/prepare_training.py" "$@"
    ;;
  configure-train)
    "$PYTHON" "$ROOT/scripts/configure_training.py" "$@"
    ;;
  train-mal)
    config=""; init=""; seed=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --config) config="$2"; shift 2 ;;
        --init) init="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        *) break ;;
      esac
    done
    [[ -n "$config" && -n "$init" && -n "$seed" ]] || { usage; exit 2; }
    "$PYTHON" "$ROOT/train_mal.py" -c "$config" -t "$init" --device cuda:0 --seed "$seed" "$@"
    ;;
  train-uq)
    "$PYTHON" "$ROOT/train_uq.py" "$@"
    ;;
  *) echo "Unknown command: $command" >&2; usage >&2; exit 2 ;;
esac
