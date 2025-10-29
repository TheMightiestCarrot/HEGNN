#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: setup_hegnn_env.sh [options]

Options:
  --cpu                  Install CPU-only wheels (default: auto-detect CUDA and prefer GPU wheels).
  --recreate             Recreate the virtual environment from scratch.
  --venv-path PATH       Location for the virtual environment (default: <repo>/.venv).
  --generate-dataset     Generate a small charged N-body dataset after installing dependencies.
  --data-path PATH       Output directory for generated data (default: datasets/nbody/data_small).
  --num-train N          Training simulations to generate (default: 200).
  --num-valid N          Validation simulations to generate (default: 50).
  --num-test N           Test simulations to generate (default: 50).
  --length N             Trajectory length for train/valid sets (default: 500).
  --length-test N        Trajectory length for the test set (default: 500).
  --sample-freq N        Sampling frequency for trajectories (default: 10).
  --n-isolated N         Number of isolated bodies (default: 5).
  --n-stick N            Number of stick constraints (default: 0).
  --n-hinge N            Number of hinge constraints (default: 0).
  --seed N               RNG seed used during dataset generation (default: 43).
  --n-workers N          Parallel workers for dataset generation (default: 4).
  -h, --help             Show this message.

Examples:
  # Create (or reuse) .venv with CUDA wheels and build the small dataset
  ./scripts/setup_hegnn_env.sh --generate-dataset

  # Force CPU wheels and place the venv elsewhere
  ./scripts/setup_hegnn_env.sh --cpu --venv-path /tmp/hegnn-venv
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"
USE_CUDA=1
RECREATE=0
GEN_DATA=0
DATA_PATH="${ROOT_DIR}/datasets/nbody/data_small"
NUM_TRAIN=200
NUM_VALID=50
NUM_TEST=50
LENGTH=500
LENGTH_TEST=500
SAMPLE_FREQ=10
N_ISOLATED=5
N_STICK=0
N_HINGE=0
SEED=43
N_WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      USE_CUDA=0
      ;;
    --recreate)
      RECREATE=1
      ;;
    --venv-path)
      shift
      VENV_PATH="$1"
      ;;
    --generate-dataset)
      GEN_DATA=1
      ;;
    --data-path)
      shift
      DATA_PATH="$1"
      ;;
    --num-train)
      shift
      NUM_TRAIN="$1"
      ;;
    --num-valid)
      shift
      NUM_VALID="$1"
      ;;
    --num-test)
      shift
      NUM_TEST="$1"
      ;;
    --length)
      shift
      LENGTH="$1"
      ;;
    --length-test)
      shift
      LENGTH_TEST="$1"
      ;;
    --sample-freq)
      shift
      SAMPLE_FREQ="$1"
      ;;
    --n-isolated)
      shift
      N_ISOLATED="$1"
      ;;
    --n-stick)
      shift
      N_STICK="$1"
      ;;
    --n-hinge)
      shift
      N_HINGE="$1"
      ;;
    --seed)
      shift
      SEED="$1"
      ;;
    --n-workers)
      shift
      N_WORKERS="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found in PATH." >&2
  exit 1
fi

# Auto-disable CUDA wheels if nvidia-smi is absent and --cpu wasn't supplied.
if [[ $USE_CUDA -eq 1 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not detected; falling back to CPU wheels." >&2
  USE_CUDA=0
fi

if [[ $USE_CUDA -eq 1 ]]; then
  TORCH_TAG="2.1.0+cu118"
  TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
  DGL_REPO="https://data.dgl.ai/wheels/cu118/repo.html"
else
  TORCH_TAG="2.1.0+cpu"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cpu.html"
  DGL_REPO="https://data.dgl.ai/wheels/repo.html"
fi

if [[ $RECREATE -eq 1 && -d "$VENV_PATH" ]]; then
  echo "[setup] Removing existing virtual environment at $VENV_PATH"
  rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "[setup] Creating virtual environment at $VENV_PATH"
  python3 -m venv "$VENV_PATH"
else
  echo "[setup] Re-using virtual environment at $VENV_PATH"
fi

PYTHON="${VENV_PATH}/bin/python"
PIP="${VENV_PATH}/bin/pip"

echo "[setup] Upgrading pip/setuptools/wheel"
"$PIP" install --upgrade pip setuptools wheel

echo "[setup] Installing torch (${TORCH_TAG})"
"$PIP" install "torch==${TORCH_TAG}" --index-url "$TORCH_INDEX"

echo "[setup] Installing project requirements"
PIP_ARGS=(
  --extra-index-url "$TORCH_INDEX"
  -f "$PYG_URL"
)
PIP_ARGS+=( -f "$DGL_REPO" )
"$PIP" install "${PIP_ARGS[@]}" -r "${ROOT_DIR}/requirements.txt"

echo "[setup] Ensuring DGL wheel matches selected backend"
"$PIP" install --force-reinstall "dgl==1.1.3" -f "$DGL_REPO"

if [[ $GEN_DATA -eq 1 ]]; then
  echo "[setup] Generating charged N-body dataset at $DATA_PATH"
  mkdir -p "$DATA_PATH"
  "$PYTHON" "${ROOT_DIR}/datasets/nbody/datagen/generate_dataset.py" \
    --num-train "$NUM_TRAIN" \
    --num-valid "$NUM_VALID" \
    --num-test "$NUM_TEST" \
    --length "$LENGTH" \
    --length_test "$LENGTH_TEST" \
    --sample-freq "$SAMPLE_FREQ" \
    --n_isolated "$N_ISOLATED" \
    --n_stick "$N_STICK" \
    --n_hinge "$N_HINGE" \
    --seed "$SEED" \
    --n_workers "$N_WORKERS" \
    --path "$DATA_PATH"
fi

echo "[setup] Done. Activate with: source \"${VENV_PATH}/bin/activate\""
