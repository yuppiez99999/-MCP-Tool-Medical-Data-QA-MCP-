#!/usr/bin/env bash
# ============================================================
# deploy_main.sh — 超算登录节点一键部署脚本（2026-06 修正版）
# 功能：
#    1. 登录节点检查
#    2. 上传项目目录（不含数据集）到超算
#    3. 创建 Python 虚拟环境并安装依赖
#    4. 检查数据集是否存在（如缺失会提示上传）
#    5. 提交 SLURM 批量估值作业
#
# 使用方法（本地 Linux / macOS 或 WSL）：
#   bash 18-医疗AI模型系统/hpc/deploy_main.sh
#
# Windows 用户推荐直接用：
#   powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass `
#       -File "e:\各种PY程序\18-医疗AI模型系统\hpc\build_and_upload.ps1"
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/deploy_config.sh"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[ERROR] 找不到配置文件: ${CONFIG_FILE}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

SSH_OPTS=(-i "${HPC_SSH_KEY}" -p "${HPC_PORT}"
          -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null
          -o LogLevel=ERROR
          -o IdentitiesOnly=yes)

run_hpc() {
    # 在超算登录节点执行一条命令，把 stdout 原样返回
    ssh "${SSH_OPTS[@]}" "${HPC_USER}@${HPC_HOST}" "$@"
}

echo "=================================="
echo "  医疗AI模型系统 — 超算中心部署"
echo "  HPC: ${HPC_USER}@${HPC_HOST}"
echo "  项目目录: ${HPC_PROJECT_ROOT}"
echo "=================================="

# ---------- 1. 连接检查 ----------
echo "[step 1/5] 检查超算登录节点连接 ..."
if ! run_hpc "echo '登录节点 OK: '$(hostname)' ; uname -a ; echo python: $(python3 --version 2>&1)'"; then
    echo "[ERROR] 无法登录超算节点。请检查 SSH Key / 用户名 / 主机名 / 网络。" >&2
    exit 1
fi

# ---------- 2. 上传项目到超算 ----------
echo "[step 2/5] 上传项目代码到超算 ..."
run_hpc "mkdir -p '${HPC_PROJECT_ROOT}' '${HPC_OUTPUT_ROOT}' '${HPC_LOG_ROOT}'"

LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARBALL="$(mktemp -t healthcare_hpc.XXXXXX.tar.gz)"
echo "  - 本地打包 ${LOCAL_PROJECT_ROOT} -> ${TARBALL}"
tar -czf "${TARBALL}" \
    --exclude='outputs' \
    --exclude='outputs_hpc' \
    --exclude='logs_hpc' \
    --exclude='env_py311' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='*.log' \
    --exclude='*.tar.gz' \
    -C "$(dirname "${LOCAL_PROJECT_ROOT}")" \
    "$(basename "${LOCAL_PROJECT_ROOT}")"

echo "  - SCP 上传到超算 ..."
scp -P "${HPC_PORT}" -i "${HPC_SSH_KEY}" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${TARBALL}" "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_ROOT}/"

echo "  - 超算端解压 ..."
TARBALL_BASENAME="$(basename "${TARBALL}")"
run_hpc "cd '${HPC_PROJECT_ROOT}' && tar -xzf '${TARBALL_BASENAME}' --strip-components=1 && rm -f '${TARBALL_BASENAME}'"
rm -f "${TARBALL}"
echo "  - 上传完成"

# ---------- 3. Python 虚拟环境 + 依赖 ----------
echo "[step 3/5] 构建 Python 虚拟环境 + 安装依赖 ..."
# 不带引号的 heredoc：本地 bash 展开 ${HPC_PROJECT_ROOT}
run_hpc bash <<HPCSH
set -euo pipefail
cd "${HPC_PROJECT_ROOT}"
if [ ! -d env_py311 ]; then
    python3 -m venv env_py311
fi
source env_py311/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || pip install -r requirements.txt
python -c "import pandas,numpy,sklearn,yaml; print('python env OK')"
HPCSH

# ---------- 4. 数据集检查 ----------
echo "[step 4/5] 检查数据集是否就位 ..."
DATASET_OK=$(run_hpc "[ -f '${HPC_DATASET}' ] && echo 1 || echo 0")
if [ "${DATASET_OK}" = "1" ]; then
    SIZE=$(run_hpc "du -m '${HPC_DATASET}' | cut -f1")
    echo "  - 数据集已就位: ${HPC_DATASET} (${SIZE} MB)"
else
    echo "  [WARN] 数据集尚未上传: ${HPC_DATASET}"
    echo "  请从本地路径上传数据集，例如："
    echo "    scp -i ${HPC_SSH_KEY} -P ${HPC_PORT} \\"
    echo "        /path/to/healthcare_token_A_B_100M.csv \\"
    echo "        ${HPC_USER}@${HPC_HOST}:${HPC_DATASET}"
    echo "  （未上传数据集时，worker 会生成伪数据用于性能压测）"
fi

# ---------- 5. 提交 SLURM 作业 ----------
echo "[step 5/5] 提交 SLURM 批量估值作业 ..."
JOB_ID=$(run_hpc "cd '${HPC_PROJECT_ROOT}' && sbatch --parsable hpc/jobs/token_valuation.slurm")
echo "==> SLURM 作业已提交: JOB_ID=${JOB_ID}"

echo ""
echo "=== 部署完成 ==="
echo "  - 查看作业队列:   ssh ${HPC_USER}@${HPC_HOST} \"squeue -u ${HPC_USER}\""
echo "  - 取消作业:        ssh ${HPC_USER}@${HPC_HOST} \"scancel ${JOB_ID}\""
echo "  - 查看输出目录:    ssh ${HPC_USER}@${HPC_HOST} \"ls -lh ${HPC_OUTPUT_ROOT}\""
echo "  - 查看日志:        ssh ${HPC_USER}@${HPC_HOST} \"ls -lh ${HPC_LOG_ROOT}\""
echo "  - 最新 30 行日志:  ssh ${HPC_USER}@${HPC_HOST} \"tail -n 30 ${HPC_LOG_ROOT}/slurm_${JOB_ID}_*.err\""
echo "  - 汇总报告:        ssh ${HPC_USER}@${HPC_HOST} \"cd ${HPC_PROJECT_ROOT} && source env_py311/bin/activate && python hpc/aggregate_hpc_reports.py --dir outputs_hpc\""
