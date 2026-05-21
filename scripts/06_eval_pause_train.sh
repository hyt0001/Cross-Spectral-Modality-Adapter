#!/usr/bin/env bash
# scripts/06_eval_pause_train.sh
#
# 单卡场景：暂停正在运行的训练 → mAP 评估 → 恢复训练。
# 默认评估 vis_every 已落盘的 epoch_0000/0010/0020/0030（对应已完成 1/11/21/31 轮）。
#
# 关于「前 35 个 epoch（0–34）」：
#   当前进程未写入 epoch_0034.pt（vis_every=10）。
#   磁盘上最接近的是 epoch_0030.pt（第 31 轮结束，差 4 个 epoch）。
#   训练代码已支持每 epoch 写 latest.pt；若需精确 ep34，请在 ep35 结束后
#   用新版训练重启，或等 ep40 的 epoch_0040.pt。
#
# 用法：
#   bash scripts/06_eval_pause_train.sh
#   CKPTS="epoch_0030.pt" bash scripts/06_eval_pause_train.sh
#   TRAIN_PID=11539 bash scripts/06_eval_pause_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

CKPT_DIR="${PROJECT_ROOT}/outputs_csma/ckpt"
LOG_DIR="${PROJECT_ROOT}/outputs_csma/logs"
mkdir -p "${LOG_DIR}"

# 默认评估本轮 phase0 已保存的 4 个里程碑
CKPTS="${CKPTS:-epoch_0000.pt epoch_0010.pt epoch_0020.pt epoch_0030.pt}"

find_train_pid() {
    if [[ -n "${TRAIN_PID:-}" ]]; then
        echo "${TRAIN_PID}"
        return
    fi
    # 必须暂停实际占用 GPU 的 python 进程，不能用 nohup 外层 bash 的 pid
    pgrep -f "^python3 -m src.train_csma" | head -1 || true
}

stop_train_tree() {
    local root="$1"
    kill -STOP "${root}" 2>/dev/null || true
    while read -r child; do
        [[ -n "${child}" ]] && kill -STOP "${child}" 2>/dev/null || true
    done < <(pgrep -P "${root}" 2>/dev/null || true)
}

cont_train_tree() {
    local root="$1"
    kill -CONT "${root}" 2>/dev/null || true
    while read -r child; do
        [[ -n "${child}" ]] && kill -CONT "${child}" 2>/dev/null || true
    done < <(pgrep -P "${root}" 2>/dev/null || true)
}

TRAIN_PID="$(find_train_pid)"
if [[ -z "${TRAIN_PID}" ]]; then
    echo "[WARN] 未找到训练进程，将直接评估（不暂停）"
    PAUSE_TRAIN=0
else
    PAUSE_TRAIN=1
    echo "[06] 训练进程 PID=${TRAIN_PID}，评估前将 SIGSTOP 暂停"
fi

if [[ "${PAUSE_TRAIN}" -eq 1 ]]; then
    stop_train_tree "${TRAIN_PID}"
    sleep 3
    echo "[06] 训练已暂停 PID=${TRAIN_PID} $(date '+%H:%M:%S')"
fi

cleanup() {
    if [[ "${PAUSE_TRAIN}" -eq 1 ]]; then
        cont_train_tree "${TRAIN_PID}"
        echo "[06] 训练已恢复 PID=${TRAIN_PID} $(date '+%H:%M:%S')"
    fi
}
trap cleanup EXIT

SUMMARY="${LOG_DIR}/eval_epochs0-34_summary.txt"
echo "eval_pause_train  $(date -Iseconds)" > "${SUMMARY}"
echo "train_pid=${TRAIN_PID:-none}  ckpts=${CKPTS}" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"

for ckpt_name in ${CKPTS}; do
    ckpt_path="${CKPT_DIR}/${ckpt_name}"
    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[SKIP] 不存在: ${ckpt_path}" | tee -a "${SUMMARY}"
        continue
    fi
    echo "========================================================"
    echo " 评估 ${ckpt_name}  开始 $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"
    if CKPT="${ckpt_path}" bash "${SCRIPT_DIR}/02_eval.sh"; then
        base="${ckpt_name%.pt}"
        json="${LOG_DIR}/eval_${base}.json"
        if [[ -f "${json}" ]]; then
            conda run --no-capture-output -n RGBtest python3 -c "
import json
r = json.load(open('${json}'))
print('${ckpt_name}: mAP@0.5={:.4f}  person={:.4f}  car={:.4f}'.format(
    r.get('map_50',0), r.get('ap_person',0), r.get('ap_car',0)))
" | tee -a "${SUMMARY}"
        fi
    else
        echo "[FAIL] ${ckpt_name}" | tee -a "${SUMMARY}"
    fi
done

echo ""
echo "[06] 汇总已写入: ${SUMMARY}"
cat "${SUMMARY}"
