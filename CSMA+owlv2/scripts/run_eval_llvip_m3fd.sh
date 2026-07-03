#!/bin/bash
# 在 LLVIP / M3FD 上评估 CSMA checkpoint（threshold=0.2）
#
# 用法：
#   CKPT=outputs_xxx/ckpt/epoch_02_ema.pt bash scripts/run_eval_llvip_m3fd.sh
#
# 可选环境变量：
#   CKPT          checkpoint 路径（必填）
#   OUT_DIR       结果 JSON 输出目录，默认与 ckpt 同实验目录下的 logs/
#   BATCH_SIZE    默认 8
#   NUM_WORKERS   默认 4

set -euo pipefail
cd "$(dirname "$0")/.."

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CKPT="${CKPT:?请设置 CKPT，例如 outputs_xxx/ckpt/best.pt}"
OUT_DIR="${OUT_DIR:-$(dirname "$(dirname "$CKPT")")/logs}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
mkdir -p "${OUT_DIR}"

echo "=== Eval LLVIP ==="
python -m src.eval_csma \
  --ckpt "${CKPT}" \
  --dataset llvip \
  --data-root /root/autodl-tmp/LLVIP \
  --split test \
  --out-json "${OUT_DIR}/llvip_test.json" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --threshold 0.2

echo
echo "=== Eval M3FD ==="
python -m src.eval_csma \
  --ckpt "${CKPT}" \
  --dataset m3fd \
  --data-root /root/autodl-tmp/M3FD \
  --split test \
  --ann-file annotations/instances_default.json \
  --text-labels person car \
  --out-json "${OUT_DIR}/m3fd_test.json" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --threshold 0.2

echo
echo "=== 完成 ==="
echo "LLVIP: ${OUT_DIR}/llvip_test.json"
echo "M3FD:  ${OUT_DIR}/m3fd_test.json"
