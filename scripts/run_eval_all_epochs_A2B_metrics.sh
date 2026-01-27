#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run `eval_all_epochs_A2B_metrics.py` with convenient defaults.
# You typically only need to provide:
#   - --test-a : your testA folder (or CycleGAN root containing testA/)
#   - --real-b : the matching realB folder
#
# Example:
#   bash UGVSM/3D_flow_consistent_UVCGANv2_vHE/scripts/run_eval_all_epochs_A2B_metrics.sh \
#     --test-a "/home/durrlab/Desktop/Anthony/data/XYZ/testA" \
#     --real-b "/home/durrlab/Desktop/Anthony/data/XYZ/testB" \
#     --split test \
#     --batch-size 1 \
#     --resume

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/eval_all_epochs_A2B_metrics.py"

# Default to the checkpoints path you referenced in your prompt; override with --checkpoints-dir.
DEFAULT_CHECKPOINTS_DIR="/home/durrlab/Desktop/Anthony/UGVSM/3D_flow_consistent_UVCGANv2_vHE/outdir/20260124_BIT2HE_normal_duodenum_only_crypts_3DFlow/20260124_duodenum_only_crypts_3DFlow_zspacing=2slices_lamsub=0p0_lamemb=0p0_lamSty=1p0/model_m(uvcgan2_3D_stylefusion)_d(basic)_g(vit-modnet)_uvcgan2-bn_(False:10.0:0.01:5e-05)/checkpoints"

CHECKPOINTS_DIR="${DEFAULT_CHECKPOINTS_DIR}"
TEST_A=""
REAL_B=""
OUTPUT_DIR=""
SPLIT="test"
BATCH_SIZE="1"
N_EVAL=""
EPOCHS=""
DATASET_NAME="cyclegan"
Z_SPACING="1"
SAMPLE_BASENAME=""
ALLOW_MISSING_METRICS="0"
RESIZE_TO_REAL="0"
RESUME="0"

usage() {
  cat <<'EOF'
Usage:
  run_eval_all_epochs_A2B_metrics.sh --test-a PATH --real-b PATH [options]

Required:
  --test-a PATH           Path to testA images OR CycleGAN root containing testA/
  --real-b PATH           Path to realB images (paired by basename)

Options:
  --checkpoints-dir PATH  Path to model checkpoints/ (default is hardcoded in this script)
  --output-dir PATH       Where to write outputs (default: <model_dir>/eval_all_epochs_metrics)
  --split {train,test,val} (default: test)
  --batch-size INT        (default: 1)
  -n, --n-eval INT        Limit number of images translated (default: all)
  --epochs SPEC           "10,20,30" or "10:100:10" (default: all available)
  --dataset-name NAME     "cyclegan" or "adjacent-z-pairs" (default: cyclegan)
  --z-spacing INT         Only for adjacent-z-pairs (default: 1)
  --sample-basename NAME  Basename (no extension) of the sample image to save each epoch
  --allow-missing-metrics Write NaN for metrics whose deps are missing (lpips / torch-fidelity / clean-fid)
  --resize-to-real        If shapes differ, resize fake->real for PSNR/SSIM/LPIPS
  --resume                Skip epochs already present in metrics_by_epoch.txt

Environment:
  CUDA_VISIBLE_DEVICES=0  Choose GPU (passed through to python)
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints-dir) CHECKPOINTS_DIR="${2:-}"; shift 2 ;;
    --test-a) TEST_A="${2:-}"; shift 2 ;;
    --real-b) REAL_B="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --split) SPLIT="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    -n|--n-eval) N_EVAL="${2:-}"; shift 2 ;;
    --epochs) EPOCHS="${2:-}"; shift 2 ;;
    --dataset-name) DATASET_NAME="${2:-}"; shift 2 ;;
    --z-spacing) Z_SPACING="${2:-}"; shift 2 ;;
    --sample-basename) SAMPLE_BASENAME="${2:-}"; shift 2 ;;
    --allow-missing-metrics) ALLOW_MISSING_METRICS="1"; shift 1 ;;
    --resize-to-real) RESIZE_TO_REAL="1"; shift 1 ;;
    --resume) RESUME="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${TEST_A}" || -z "${REAL_B}" ]]; then
  echo "Error: --test-a and --real-b are required." >&2
  usage
  exit 2
fi

cmd=(python3 "${PY_SCRIPT}"
  --checkpoints-dir "${CHECKPOINTS_DIR}"
  --test-a "${TEST_A}"
  --real-b "${REAL_B}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --dataset-name "${DATASET_NAME}"
  --z-spacing "${Z_SPACING}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${N_EVAL}" ]]; then
  cmd+=(--n-eval "${N_EVAL}")
fi
if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi
if [[ -n "${SAMPLE_BASENAME}" ]]; then
  cmd+=(--sample-basename "${SAMPLE_BASENAME}")
fi
if [[ "${ALLOW_MISSING_METRICS}" == "1" ]]; then
  cmd+=(--allow-missing-metrics)
fi
if [[ "${RESIZE_TO_REAL}" == "1" ]]; then
  cmd+=(--resize-to-real)
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi

echo "[INFO] Running:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"

