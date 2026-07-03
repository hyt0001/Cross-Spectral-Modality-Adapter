#!/usr/bin/env bash
# scripts/07_precheck_ablation.sh
#
# 全量训练前的轻量预检（Ablation B-2 / C-1 / C-2 / C-3）
#
# 目的：
#   smoke test（2 epoch × 20 step）只验证"不炸"。
#   本脚本用完整数据做短程训练，确认：
#     1. 所有分支 loss 均正常（非 NaN / Inf）
#     2. C-3 (gmm_single_b) 没有触发 GMM fallback
#     3. 每个分支生成 ckpt/latest.pt 和 ckpt/csma_last.pt
#     4. B-2 (mean_proto) 需要先有 mean_proto.pt
#
# 用法（轻量，默认）：
#   bash scripts/07_precheck_ablation.sh
#
# 用法（完整预检）：
#   EPOCHS=5 MAX_STEPS=-1 bash scripts/07_precheck_ablation.sh
#
# 用法（跳过 B-2）：
#   SKIP_B2=1 bash scripts/07_precheck_ablation.sh
#
# 通过标准：
#   - 所有组 exit code = 0
#   - 所有组生成 ckpt/latest.pt 和 ckpt/csma_last.pt
#   - C-3 日志中 gmm_single_b fallback 次数 = 0
#   - 日志中无真正的 loss=nan 或 loss=inf（非 info/inference 误匹配）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 可覆盖的参数 ───────────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/FLIR_License/train}"
MEAN_PROTO_PATH="${MEAN_PROTO_PATH:-${PROJECT_ROOT}/outputs_csma/mean_proto.pt}"
DEBUG_DET_LOSS="${DEBUG_DET_LOSS:-0}"
SKIP_B2="${SKIP_B2:-0}"

# 独立输出目录，不写入正式全量目录
PRECHECK_ROOT="${PROJECT_ROOT}/outputs_precheck"

# ── 启动信息 ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Ablation 预检脚本"
echo " EPOCHS         = ${EPOCHS}"
echo " MAX_STEPS      = ${MAX_STEPS}"
echo " BATCH_SIZE     = ${BATCH_SIZE}"
echo " DATA_ROOT      = ${DATA_ROOT}"
echo " DEBUG_DET_LOSS = ${DEBUG_DET_LOSS}"
echo " PRECHECK_ROOT  = ${PRECHECK_ROOT}"
echo "   c1  => ${PRECHECK_ROOT}/c1_random_mask"
echo "   c2  => ${PRECHECK_ROOT}/c2_fixed_threshold"
echo "   c3  => ${PRECHECK_ROOT}/c3_gmm_single_b"
echo "   b2  => ${PRECHECK_ROOT}/b2_mean_proto"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "[ERROR] 数据目录不存在: ${DATA_ROOT}"
    exit 1
fi

# ── 运行单个分支 ───────────────────────────────────────────────────────────────
_run() {
    local name="$1"; shift
    local out="${PRECHECK_ROOT}/${name}"
    mkdir -p "${out}/logs"
    echo ""
    echo "---- [预检] ${name}  out=${out} ----"

    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/autodl-tmp/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DEBUG_DET_LOSS="${DEBUG_DET_LOSS}" \
    python3 -m src.train_csma \
        --dataset    flir_v1 \
        --data-root  "${DATA_ROOT}" \
        --out-dir    "${out}" \
        --epochs     "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --max-steps  "${MAX_STEPS}" \
        "$@" \
    2>&1 | tee "${out}/logs/precheck.log"

    local code=${PIPESTATUS[0]}
    if [[ $code -ne 0 ]]; then
        echo "[FAIL] ${name} 预检失败（exit=${code}）"
        return 1
    fi
    echo "[OK] ${name} exit=0"
    return 0
}

# ── 运行各消融分支 ─────────────────────────────────────────────────────────────
_run "c1_random_mask"     --cmss-ablation-mode random_mask
_run "c2_fixed_threshold" --cmss-ablation-mode fixed_threshold
_run "c3_gmm_single_b"    --cmss-ablation-mode gmm_single_b

if [[ "${SKIP_B2}" == "1" ]]; then
    echo ""
    echo "[跳过] B-2 预检（SKIP_B2=1，mean_proto.pt 需先运行 scripts/06_ablation_b2.sh Step1）"
elif [[ ! -f "${MEAN_PROTO_PATH}" ]]; then
    echo ""
    echo "[跳过] B-2 预检（mean_proto.pt 未就绪: ${MEAN_PROTO_PATH}）"
    echo "        请先运行: bash scripts/06_ablation_b2.sh（仅 Step 1）"
else
    _run "b2_mean_proto" \
        --variant mean_proto \
        --mean-proto-path "${MEAN_PROTO_PATH}"
fi

# ── 验证 1：C-3 GMM fallback ──────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " [验证 1] C-3 GMM fallback 检查..."
c3_log="${PRECHECK_ROOT}/c3_gmm_single_b/logs/precheck.log"
if [[ -f "${c3_log}" ]]; then
    fallback_count=$(grep -c "gmm_single_b fallback to fixed threshold" "${c3_log}" || true)
    if [[ "${fallback_count}" -gt 0 ]]; then
        echo "[FAIL] C-3 日志中出现 ${fallback_count} 次 GMM fallback！"
        echo "       C-3 本应全程 GMM Stage B，请检查 collect_cmss_values 数据量。"
        echo "       相关行："
        grep "gmm_single_b fallback" "${c3_log}" || true
        exit 1
    else
        echo "[OK] C-3 fallback_count=0，GMM 全程正常"
    fi
else
    echo "[SKIP] C-3 日志不存在，跳过 fallback 检查"
fi

# ── 验证 2：NaN/Inf（用 Python 做精确匹配，避免误匹配 info/inference）──────────
echo ""
echo " [验证 2] NaN/Inf 精确扫描（Python）..."
python3 - <<'PYEOF'
import re, sys, os, glob

precheck_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs_precheck"
)

# 匹配 "loss=nan"、"loss=inf"、"loss= nan" 等赋值形式，排除 info / inference
# 同时捕获独立单词 nan/inf 出现在数值上下文中
PAT_NAN  = re.compile(r'(?<![a-zA-Z_])nan(?![a-zA-Z_])', re.IGNORECASE)
PAT_INF  = re.compile(r'(?<![a-zA-Z_])inf(?![a-zA-Z_o])', re.IGNORECASE)
# 白名单：含 info、inference、inferred、infimum、initialize、InfoNCE 等词的行直接跳过
WHITELIST = re.compile(r'\binfo\b|\binference\b|\binferred\b|\binitiali', re.IGNORECASE)

failed = False
names = ["c1_random_mask", "c2_fixed_threshold", "c3_gmm_single_b", "b2_mean_proto"]
for name in names:
    log = os.path.join(precheck_root, name, "logs", "precheck.log")
    if not os.path.exists(log):
        print(f"[SKIP] {name}: 日志不存在")
        continue
    nan_lines, inf_lines = [], []
    with open(log, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if WHITELIST.search(line):
                continue
            if PAT_NAN.search(line):
                nan_lines.append((lineno, line.rstrip()))
            if PAT_INF.search(line):
                inf_lines.append((lineno, line.rstrip()))
    if nan_lines or inf_lines:
        print(f"[FAIL] {name}: nan命中={len(nan_lines)}, inf命中={len(inf_lines)}")
        for lineno, txt in (nan_lines + inf_lines)[:5]:
            print(f"       L{lineno}: {txt}")
        failed = True
    else:
        print(f"[OK]   {name}: nan=0, inf=0")

sys.exit(1 if failed else 0)
PYEOF

# ── 验证 3：checkpoint 文件存在 ───────────────────────────────────────────────
echo ""
echo " [验证 3] checkpoint 文件检查..."
_check_ckpt() {
    local name="$1"
    local out="${PRECHECK_ROOT}/${name}"
    local latest="${out}/ckpt/latest.pt"
    local last="${out}/ckpt/csma_last.pt"
    local ok=1
    [[ -f "${latest}" ]] || { echo "[FAIL] ${name}: ckpt/latest.pt 不存在"; ok=0; }
    [[ -f "${last}"   ]] || { echo "[FAIL] ${name}: ckpt/csma_last.pt 不存在"; ok=0; }
    [[ "${ok}" == "1" ]] && echo "[OK]   ${name}: latest.pt + csma_last.pt ✓"
    return $((1 - ok))
}

ckpt_ok=0
_check_ckpt "c1_random_mask"     || ckpt_ok=1
_check_ckpt "c2_fixed_threshold" || ckpt_ok=1
_check_ckpt "c3_gmm_single_b"    || ckpt_ok=1
if [[ -f "${MEAN_PROTO_PATH}" && "${SKIP_B2}" != "1" ]]; then
    _check_ckpt "b2_mean_proto" || ckpt_ok=1
fi
[[ "${ckpt_ok}" -eq 0 ]] || exit 1

# ── 完成 ───────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " 预检完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 所有检查通过，可启动全量 100 epoch 训练："
echo "   bash scripts/05_ablation_c.sh"
echo "   bash scripts/06_ablation_b2.sh"
echo "========================================================"
