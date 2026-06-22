#!/usr/bin/env bash
# ============================================================
# run_v2_token_gen.sh — 超算端一键执行 v2 Token 生成 + 打包脚本
# 对应《超算中心Token生成操作手册_v2.md》完整流程
#
# 用法：
#   bash run_v2_token_gen.sh           # 默认 scale=5000（推荐 ~240 万条）
#   bash run_v2_token_gen.sh 20        # 小测 ~9600 条
#   bash run_v2_token_gen.sh 500       # ~24 万条
#   bash run_v2_token_gen.sh 10000     # ~480 万条（超大规模）
#
# 环境变量控制：
#   SCALE_TEST_ONLY=1  bash run_v2_token_gen.sh    # 仅跑小测，不跑大规模
#   PY_BIN=python     bash run_v2_token_gen.sh    # 强制用 python2
#
# 行为：
#   1. 确保项目目录 ~/token_project/
#   2. 复制本脚本所在目录的 gen_large.py 到项目目录
#   3. 先跑 scale=20 的小测（除非 SCALE_TEST_ONLY=1）
#   4. 后台 nohup 正式跑，定时打印心跳
#   5. 生成完成后自动打包为 北数所上架包 tar.gz
#   6. 输出所有路径、日志位置、下载提示
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HPC_USER="$(whoami)"
HOME_DIR="${HOME}"
PROJECT_DIR="${HOME_DIR}/token_project"
OUTPUT_DIR="${PROJECT_DIR}/output"
LOG_DIR="${PROJECT_DIR}/logs"
PKG_PARENT="${HOME_DIR}"
PKG_DIR="${PKG_PARENT}/北数所上架包"

# --- 参数解析 ---
SCALE="${1:-5000}"
SCALE_TEST_ONLY="${SCALE_TEST_ONLY:-0}"
PY_BIN="${PY_BIN:-}"

log() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" >&2; }

if [ -z "${PY_BIN}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY_BIN="python3"
    else
        PY_BIN="python"
    fi
fi

log "使用 python : ${PY_BIN} ($(${PY_BIN} --version 2>&1 | head -n 1))"
log "项目目录    : ${PROJECT_DIR}"
log "scale       : ${SCALE}"
log "脚本目录    : ${SCRIPT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${PROJECT_DIR}"

# --- 复制 gen_large.py 到项目目录
SRC_PY="${SCRIPT_DIR}/gen_large.py"
PROJECT_PY="${PROJECT_DIR}/gen_large.py"
if [ -f "${SRC_PY}" ]; then
    cp -f "${SRC_PY}" "${PROJECT_PY}"
    chmod +x "${PROJECT_PY}"
    log "已复制 gen_large.py"
else
    # 允许没有 gen_large.py 时使用项目目录下已有的
    if [ -f "${PROJECT_PY}" ]; then
        log "使用项目目录下现有的 gen_large.py"
    else
        echo "[ERROR] 未找到 gen_large.py" >&2
        echo "  查找路径: ${SRC_PY} 或 ${PROJECT_PY}" >&2
        exit 1
    fi
fi

cd "${PROJECT_DIR}"

# --- 小测：scale=20
log "[1/4] 自检 + 小测 (scale=20) ..."
"${PY_BIN}" gen_large.py --scale 20
log "  OK: 小测通过"

if [ "${SCALE_TEST_ONLY}" = "1" ]; then
    log "SCALE_TEST_ONLY=1，仅跑小测，退出。"
    exit 0
fi

# --- 正式生成：nohup 后台，避免 Web 终端会话退出后进程被杀
LOG_FILE="${LOG_DIR}/token_gen_$(date +%Y%m%d_%H%M%S).log"
log "[2/4] 正式生成，scale=${SCALE}，日志: ${LOG_FILE}"

cd "${PROJECT_DIR}"
nohup "${PY_BIN}" gen_large.py --scale "${SCALE}" > "${LOG_FILE}" 2>&1 &
PID=$!
log "  进程 PID=${PID}"
log "  查看日志 : tail -n 30 ${LOG_FILE}"
log "  查看进程 : ps -fp ${PID}"

# --- 轮询等待进程结束，每 60 秒打印一次心跳
log "[3/4] 等待进程结束（每 60 秒输出一次心跳）..."
WAIT_START=$(date +%s)
while kill -0 "${PID}" 2>/dev/null; do
    sleep 60
    NOW=$(date +%s)
    ELAPSED=$(( NOW - WAIT_START ))
    LAST_LINES="$(tail -n 3 "${LOG_FILE}" 2>/dev/null || true)"
    log "  已运行 ${ELAPSED} s，最新 3 行日志:"
    printf "    %s\n" "${LAST_LINES}" 2>/dev/null || true
done

# 等待文件系统 flush
sleep 2

# --- 验证生成结果
log "[4/4] 验证结果..."
echo ""
echo "=== 结果统计 ==="
HAS_CSV=0
TOTAL_LINES=0
for f in "${OUTPUT_DIR}"/*_token_*.csv; do
    if [ -f "${f}" ]; then
        HAS_CSV=1
        lines=$(wc -l < "${f}" | tr -d ' ')
        size=$(du -m "${f}" | cut -f1)
        TOTAL_LINES=$(( TOTAL_LINES + lines - 1 ))
        printf "  %-55s  %8d 行  %s MB\n" "$(basename "${f}")" "$(( lines - 1 ))" "${size}"
    fi
done
if [ "${HAS_CSV}" = "0" ]; then
    echo "[ERROR] output/ 下没有 *_token_*.csv，生成可能失败。请检查 ${LOG_FILE}" >&2
    exit 2
fi
echo "  总计: ${TOTAL_LINES} 条 token 记录"
echo ""
if ls "${OUTPUT_DIR}"/MASTER_REPORT_*.json >/dev/null 2>&1; then
    echo "=== MASTER_REPORT ==="
    cat "${OUTPUT_DIR}"/MASTER_REPORT_*.json | head -n 80
    echo ""
fi

# --- 打包为 北数所上架包/01数据文件 等目录
log "打包为北数所上架包格式..."
rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}/01数据文件"
mkdir -p "${PKG_DIR}/02合规文档"
mkdir -p "${PKG_DIR}/03卖家备案"
mkdir -p "${PKG_DIR}/04API服务"
cp -f "${OUTPUT_DIR}"/*_token_*.csv "${PKG_DIR}/01数据文件/"
if ls "${OUTPUT_DIR}"/*.json >/dev/null 2>&1; then
    cp -f "${OUTPUT_DIR}"/*.json "${PKG_DIR}/02合规文档/"
fi

# --- 写一个纯 ASCII 的 README 到 02合规文档/
cat > "${PKG_DIR}/02合规文档/README.txt" <<'EOF'
This package is the BeiJing-Data-Exchange Token generation output.
Produced on the HPC login node of the National Supercomputer Center.
Generator : python gen_large.py --scale <see MASTER_REPORT_*.json>
Timestamp : see MASTER_REPORT_*.json
EOF

TAR_PATH="${PKG_PARENT}/北数所Token_HPC生成包_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "${TAR_PATH}" -C "${PKG_PARENT}" 北数所上架包/

echo ""
echo "========================================================="
echo "  生成结束"
echo "  项目目录 : ${PROJECT_DIR}"
echo "  输出目录 : ${OUTPUT_DIR}"
echo "  日志目录 : ${LOG_DIR}"
echo "  日志文件 : ${LOG_FILE}"
echo "  上架包   : ${TAR_PATH}"
echo "  文件大小 : $(du -mh "${TAR_PATH}" | cut -f1)"
echo "========================================================="
echo ""
echo "=== 下载方式 ==="
echo "  1) 浏览器登录 https://www.scnet.cn (用户: ${HPC_USER})"
echo "     在「文件管理」中浏览到以下路径并下载 tar.gz:"
echo "       $(echo "${TAR_PATH}")"
echo ""
echo "  2) 如本地 PowerShell 可解析超算主机名 (需先解决 DNS/SSH 通道)"
echo "     scp ${HPC_USER}@<HPC_HOST>:${TAR_PATH} ./"
echo ""
echo "=== 快速自检命令 ==="
echo "  ls -lh ${OUTPUT_DIR}"
echo "  cat ${OUTPUT_DIR}/MASTER_REPORT_*.json"
echo "  tail -n 30 ${LOG_FILE}"
echo ""
