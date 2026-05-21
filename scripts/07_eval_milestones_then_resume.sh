#!/usr/bin/env bash
# 训练已 SIGSTOP 后：依次评估里程碑 ckpt，完成后 SIGCONT 恢复训练。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_PID="${TRAIN_PID:-$(pgrep -f '^python3 -m src.train_csma' | head -1)}"
CKPTS="${CKPTS:-epoch_0000.pt epoch_0010.pt epoch_0020.pt epoch_0030.pt}"
LOG="${PROJECT_ROOT}/outputs_csma/logs/eval_milestones_then_resume.log"

exec >>"${LOG}" 2>&1
echo "=== $(date -Iseconds) TRAIN_PID=${TRAIN_PID} ==="

for ckpt_name in ${CKPTS}; do
    ckpt="${PROJECT_ROOT}/outputs_csma/ckpt/${ckpt_name}"
    [[ -f "${ckpt}" ]] || { echo "[SKIP] ${ckpt}"; continue; }
    echo "--- eval ${ckpt_name} ---"
    CKPT="${ckpt}" bash "${PROJECT_ROOT}/scripts/02_eval.sh" || echo "[FAIL] ${ckpt_name}"
done

if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill -CONT "${TRAIN_PID}" 2>/dev/null || true
    for c in $(pgrep -P "${TRAIN_PID}" 2>/dev/null || true); do kill -CONT "${c}" 2>/dev/null || true; done
    echo "[OK] 训练已恢复 PID=${TRAIN_PID}"
fi
echo "=== done $(date -Iseconds) ==="
