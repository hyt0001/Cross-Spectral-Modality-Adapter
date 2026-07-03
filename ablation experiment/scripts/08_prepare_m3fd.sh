#!/usr/bin/env bash
# scripts/08_prepare_m3fd.sh
#
# M3FD 数据集准备流水线
#
# 步骤：
#   Step 1. 解压 ir.zip 和 vi.zip
#   Step 2. 整理 YOLO 标注文件（从 zip 内 .txt 或 labels/ 目录）
#   Step 3. 运行 convert_m3fd_to_coco.py（YOLO → COCO JSON，自动 train/val 拆分）
#   Step 4. 打印目录结构摘要
#
# 前置条件（用户已手动执行下载命令）：
#   conda run -n RGBtest huggingface-cli download \
#       --repo-type dataset --local-dir M3FD/hf_raw \
#       nonameplease/M3FD_Detecion ir.zip vi.zip \
#       annotations/instances_default.json classes.txt
#
# 用法：
#   cd /root/autodl-tmp/Cross-Spectral-Modality-Adapter
#   bash scripts/08_prepare_m3fd.sh
#
# 可覆盖的环境变量：
#   RAW_DIR          HuggingFace 原始下载目录（默认 M3FD/hf_raw）
#   OUT_DIR          整理后数据根目录（默认 M3FD）
#   TRAIN_RATIO      train 比例（默认 0.8）
#   SEED             随机种子（默认 42）
#   KEEP_BUS         若为 1，Bus 归并到 car（默认 0）
#   KEEP_TRUCK       若为 1，Truck 归并到 car（默认 0）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RAW_DIR="${RAW_DIR:-${PROJECT_ROOT}/M3FD/hf_raw}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/M3FD}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SEED="${SEED:-42}"
KEEP_BUS="${KEEP_BUS:-0}"
KEEP_TRUCK="${KEEP_TRUCK:-0}"

echo "========================================================"
echo " M3FD 数据集准备"
echo " RAW_DIR    = ${RAW_DIR}"
echo " OUT_DIR    = ${OUT_DIR}"
echo " TRAIN_RATIO= ${TRAIN_RATIO}  SEED=${SEED}"
echo " KEEP_BUS   = ${KEEP_BUS}  KEEP_TRUCK=${KEEP_TRUCK}"
echo " 开始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# ── 检查原始下载目录 ──────────────────────────────────────────────────────────
if [[ ! -d "${RAW_DIR}" ]]; then
    echo "[ERROR] 原始下载目录不存在: ${RAW_DIR}"
    echo "请先执行 huggingface-cli download 命令"
    exit 1
fi

IR_ZIP="${RAW_DIR}/ir.zip"
VI_ZIP="${RAW_DIR}/vi.zip"
CLASSES_TXT="${RAW_DIR}/classes.txt"

for f in "${IR_ZIP}" "${VI_ZIP}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] 文件不存在: ${f}"
        exit 1
    fi
done

# ── Step 1：解压 ir.zip ───────────────────────────────────────────────────────
IR_DIR="${OUT_DIR}/ir"
mkdir -p "${IR_DIR}"
echo ""
echo "[Step 1a] 解压 ir.zip → ${IR_DIR}/"
unzip -q -o "${IR_ZIP}" -d "${IR_DIR}_tmp"

# 兼容两种 zip 内部结构：
#   结构 A：zip 根目录直接是图像文件
#   结构 B：zip 内有一级子目录（如 ir/）
_flatten_dir() {
    local src="$1"
    local dst="$2"
    # 找所有图像文件，移动到 dst
    find "${src}" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
        -exec mv -n {} "${dst}/" \;
    rm -rf "${src}"
}
_flatten_dir "${IR_DIR}_tmp" "${IR_DIR}"
IR_COUNT=$(ls "${IR_DIR}" | wc -l)
echo "  IR 图像数量: ${IR_COUNT}"

# ── Step 2：解压 vi.zip ───────────────────────────────────────────────────────
VI_DIR="${OUT_DIR}/vi"
mkdir -p "${VI_DIR}"
echo ""
echo "[Step 1b] 解压 vi.zip → ${VI_DIR}/"
unzip -q -o "${VI_ZIP}" -d "${VI_DIR}_tmp"
_flatten_dir "${VI_DIR}_tmp" "${VI_DIR}"
VI_COUNT=$(ls "${VI_DIR}" | wc -l)
echo "  VI 图像数量: ${VI_COUNT}"

if [[ "${IR_COUNT}" -ne "${VI_COUNT}" ]]; then
    echo "[警告] IR 与 VI 图像数量不一致: IR=${IR_COUNT}  VI=${VI_COUNT}"
    echo "       配对时以 IR 文件为准，缺失 VI 的样本将无 L_align"
fi

# ── Step 3：整理 YOLO 标注文件 ─────────────────────────────────────────────────
LABEL_DIR="${OUT_DIR}/labels"
mkdir -p "${LABEL_DIR}"
echo ""
echo "[Step 2] 整理 YOLO 标注文件 → ${LABEL_DIR}/"

# 优先级 A：ir/ 目录内已有 .txt 文件（YOLO 格式与图像同目录）
TXT_IN_IR=$(find "${IR_DIR}" -name "*.txt" | wc -l)
if [[ "${TXT_IN_IR}" -gt 0 ]]; then
    echo "  发现 IR 目录内含 ${TXT_IN_IR} 个 .txt 文件，移动到 labels/"
    find "${IR_DIR}" -name "*.txt" -exec mv {} "${LABEL_DIR}/" \;

# 优先级 B：RAW_DIR/labels/ 目录
elif [[ -d "${RAW_DIR}/labels" ]]; then
    echo "  从 ${RAW_DIR}/labels/ 复制标注文件"
    cp "${RAW_DIR}"/labels/*.txt "${LABEL_DIR}/" 2>/dev/null || true

# 优先级 C：检查 instances_default.json 是否已是 COCO 格式
elif [[ -f "${RAW_DIR}/annotations/instances_default.json" ]]; then
    echo ""
    echo "[提示] 未找到 YOLO .txt 文件。"
    echo "       检测到 annotations/instances_default.json，"
    echo "       该文件可能已是 COCO 格式（CVAT 导出）。"
    echo "       正在验证..."
    python3 - <<'PYEOF'
import json, sys
try:
    with open("${RAW_DIR}/annotations/instances_default.json") as f:
        data = json.load(f)
    required = {"images", "annotations", "categories"}
    if required.issubset(data.keys()):
        print("  [OK] instances_default.json 是标准 COCO 格式，无需转换")
        print("       建议直接：")
        print("         cp annotations/instances_default.json M3FD/annotations_coco.json")
        print("       然后按需分割 train/val，跳过 convert_m3fd_to_coco.py")
    else:
        print("  [?]  文件格式未知，缺少字段: " + str(required - data.keys()))
except Exception as e:
    print(f"  [!]  读取失败: {e}")
PYEOF
    echo "[INFO] 如需强制 YOLO 转换，请手动解压 zip 内的 .txt 文件到 ${LABEL_DIR}/"
    exit 0
else
    echo "[ERROR] 未找到任何 YOLO .txt 标注文件"
    echo "        请确认 ir.zip 内含 .txt 文件，或手动解压标注到 ${LABEL_DIR}/"
    exit 1
fi

LABEL_COUNT=$(ls "${LABEL_DIR}" | wc -l)
echo "  标注文件数量: ${LABEL_COUNT}"

# ── Step 4：YOLO → COCO 转换 ──────────────────────────────────────────────────
ANN_TRAIN="${OUT_DIR}/train/annotations_coco.json"
ANN_VAL="${OUT_DIR}/val/annotations_coco.json"
TRAIN_IR="${OUT_DIR}/train/ir"
TRAIN_VI="${OUT_DIR}/train/vi"
VAL_IR="${OUT_DIR}/val/ir"
VAL_VI="${OUT_DIR}/val/vi"

# 构建可选参数
EXTRA_ARGS=""
[[ "${KEEP_BUS}"   == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --keep-bus"
[[ "${KEEP_TRUCK}" == "1" ]] && EXTRA_ARGS="${EXTRA_ARGS} --keep-truck"
[[ -f "${CLASSES_TXT}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --classes-txt ${CLASSES_TXT}"

echo ""
echo "[Step 3] YOLO → COCO 转换（train/val 拆分 ${TRAIN_RATIO}:$(echo "1 - ${TRAIN_RATIO}" | bc)）"
mkdir -p "${OUT_DIR}/train" "${OUT_DIR}/val"

python3 src/convert_m3fd_to_coco.py \
    --img-dir    "${IR_DIR}" \
    --label-dir  "${LABEL_DIR}" \
    --output     "${ANN_TRAIN}" \
    --val-output "${ANN_VAL}" \
    --train-ratio "${TRAIN_RATIO}" \
    --seed        "${SEED}" \
    ${EXTRA_ARGS}

# ── Step 5：按 train/val 分割移动图像 ────────────────────────────────────────
echo ""
echo "[Step 4] 按 annotations 中 file_name 分割 IR / VI 图像..."
python3 - <<PYEOF
import json, os, shutil

for split, ann_path, ir_dst, vi_dst, vi_src in [
    ("train", "${ANN_TRAIN}", "${TRAIN_IR}", "${TRAIN_VI}", "${VI_DIR}"),
    ("val",   "${ANN_VAL}",   "${VAL_IR}",  "${VAL_VI}",  "${VI_DIR}"),
]:
    os.makedirs(ir_dst, exist_ok=True)
    os.makedirs(vi_dst, exist_ok=True)
    with open(ann_path) as f:
        coco = json.load(f)
    for img in coco["images"]:
        fn = img["file_name"]
        stem = os.path.splitext(fn)[0]
        # IR
        src_ir = os.path.join("${IR_DIR}", fn)
        if os.path.exists(src_ir):
            shutil.copy2(src_ir, os.path.join(ir_dst, fn))
        # VI（同 stem，扩展名可能不同）
        for ext in [".png", ".jpg", ".jpeg"]:
            src_vi = os.path.join(vi_src, stem + ext)
            if os.path.exists(src_vi):
                shutil.copy2(src_vi, os.path.join(vi_dst, stem + ext))
                break
    print(f"  [{split}] IR={len(os.listdir(ir_dst))}  VI={len(os.listdir(vi_dst))}")
PYEOF

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " 目录结构："
echo "   ${OUT_DIR}/train/ir/           ← 红外训练图像"
echo "   ${OUT_DIR}/train/vi/           ← 可见光训练图像"
echo "   ${OUT_DIR}/train/annotations_coco.json"
echo "   ${OUT_DIR}/val/ir/"
echo "   ${OUT_DIR}/val/vi/"
echo "   ${OUT_DIR}/val/annotations_coco.json"
echo ""
echo " 下一步训练命令（在 Cross-Spectral-Modality-Adapter 项目根目录执行）："
echo "   CUDA_VISIBLE_DEVICES=0 \\"
echo "   HF_HOME=/root/autodl-tmp/hf_cache \\"
echo "   python3 -m src.train_csma \\"
echo "       --dataset m3fd \\"
echo "       --data-root ${OUT_DIR}/train \\"
echo "       --out-dir outputs_m3fd \\"
echo "       --epochs 100"
echo ""
echo " 完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
