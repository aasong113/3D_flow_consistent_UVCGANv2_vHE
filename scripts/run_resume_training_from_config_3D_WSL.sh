#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/resume_training_from_config_3D.py"

MODEL_DIR_DEFAULT="/home/durrlab-asong/Anthony/3D_flow_consistent_UVCGANv2_vHE/outdir/20260210_Inverted_Combined_BIT2HE_duodenum_crypts_lieberkuhn_Style_injection/outdir/20260211_Inverted_Combined_BIT2HE_crypts_lieberkuhn_Style_injection_Train/20260211_duodenum_only_crypts_3DFlow_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamSty=1p0/model_m(uvcgan2_3D_stylefusion)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:10.0:0.01:5e-05)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR_DEFAULT}"
CHECKPOINT_EVERY="5"
EPOCHS_OVERRIDE=""
LOG_LEVEL="DEBUG"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  bash run_resume_training_from_config_3D_WSL.sh [options]

Options:
  --model-dir PATH         Existing model directory (contains config.json/checkpoints/)
  --checkpoint-every INT   Save checkpoint every N epochs (default: 5)
  --epochs INT             Override total epochs (default: use config.json value)
  --log-level LEVEL        Logging level for training (default: DEBUG)
  --dry-run                Print resolved settings without starting training
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
    --epochs) EPOCHS_OVERRIDE="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

cmd=(
  "${PYTHON_BIN}" "${PY_SCRIPT}"
  --model-dir "${MODEL_DIR}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --log-level "${LOG_LEVEL}"
)

if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
  cmd+=(--epochs "${EPOCHS_OVERRIDE}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "[INFO] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
