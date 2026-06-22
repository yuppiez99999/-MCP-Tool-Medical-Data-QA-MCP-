#!/usr/bin/env bash
# ============================================================
# start_hpc_job.sh —— 超算端一键启动脚本（2026-06）
#
# 适用场景：
#   - 项目代码已上传到超算端
#   - 依赖（requirements.txt）已安装
#   - 想手动控制：
#       * 重新提交 SLURM 批量估值
#       * 检查虚拟环境 / 数据集 / 输出目录
#       * 生成汇总报告
#
# 用法：
#   ssh sconqw05b@scno.hpccube.com
#   cd 18-医疗AI模型系统
#   bash hpc/start_hpc_job.sh        # 提交新的 SLURM 批量作业
#   bash hpc/start_hpc_job.sh report # 作业完成后汇总报告
#   bash hpc/start_hpc_job.sh status # 查看当前队列
#   bash hpc/start_hpc_job.sh cancel <job_id>  # 取消作业
# ============================================================
set -euo pipefail

# -------- 路径 --------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLURM_SCRIPT="${PROJECT_ROOT}/hpc/jobs/token_valuation.slurm"
OUTPUT_DIR="${PROJECT_ROOT}/outputs_hpc"
LOG_DIR="${PROJECT_ROOT}/logs_hpc"
DATASET_PATH="${PROJECT_ROOT}/healthcare_token_A_B_100M.csv"
VENV_PATH="${PROJECT_ROOT}/env_py311"
WHOAMI="$(whoami)"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "  OK $*" >&2; }
warn() { echo "  WARN $*" >&2; }

action="${1:-submit}"

log "项目根目录: ${PROJECT_ROOT}"
log "当前用户    : ${WHOAMI}"
log "动作        : ${action}"

# --------- 自检 ---------
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "[ERROR] 找不到 SLURM 脚本: ${SLURM_SCRIPT}" >&2
    echo "  请确认项目已上传到超算节点。" >&2
    exit 1
fi
if [ ! -d "${VENV_PATH}" ]; then
    warn "虚拟环境不存在: ${VENV_PATH}，将自动创建"
    python3 -m venv "${VENV_PATH}"
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"
python -c "import pandas,numpy,sklearn,yaml"
ok "Python 虚拟环境 OK"

if [ -f "${DATASET_PATH}" ]; then
    SIZE_MB="$(du -m "${DATASET_PATH}" | cut -f1)"
    ok "数据集就绪: ${DATASET_PATH} (${SIZE_MB} MB)"
else
    warn "数据集缺失: ${DATASET_PATH}。Worker 会退化到伪数据，只用于性能压测。"
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

case "${action}" in
    submit)
        log "提交 SLURM 数组作业: ${SLURM_SCRIPT}"
        JOB_ID="$(cd "${PROJECT_ROOT}" && sbatch --parsable hpc/jobs/token_valuation.slurm)"
        if [ -z "${JOB_ID}" ]; then
            echo "[ERROR] sbatch 没有返回 JOB_ID，请看上方报错。" >&2
            exit 1
        fi
        echo "----------------------------------------------------"
        echo " JOB_ID = ${JOB_ID}"
        echo " 队列  : squeue -u ${WHOAMI}"
        echo " 取消  : scancel ${JOB_ID}"
        echo " 输出  : ls -lh ${OUTPUT_DIR}"
        echo " 日志  : ls -lh ${LOG_DIR}"
        echo " 最近错误 : tail -n 30 ${LOG_DIR}/slurm_${JOB_ID}_*.err"
        echo " 汇总报告 : bash $0 report"
        echo "----------------------------------------------------"
        ;;
    report)
        log "生成汇总报告（inputs: ${OUTPUT_DIR}）"
        python -u "${PROJECT_ROOT}/hpc/aggregate_hpc_reports.py" --dir "${OUTPUT_DIR}"
        ;;
    status)
        log "当前 SLURM 队列（用户 ${WHOAMI}）"
        squeue -u "${WHOAMI}" || true
        echo ""
        log "最近日志文件:"
        ls -1th "${LOG_DIR}" 2>/dev/null | head -n 6
        echo ""
        log "最近输出文件:"
        ls -1th "${OUTPUT_DIR}" 2>/dev/null | head -n 10
        ;;
    cancel)
        TARGET="${2:-}"
        if [ -z "${TARGET}" ]; then
            echo "用法: bash $0 cancel <job_id>" >&2
            exit 2
        fi
        log "取消作业 ${TARGET}"
        scancel "${TARGET}"
        sleep 1
        squeue -u "${WHOAMI}"
        ;;
    *)
        echo "未知 action: ${action}" >&2
        echo "支持: submit | report | status | cancel <job_id>" >&2
        exit 2
        ;;
esac
