#!/usr/bin/env bash
# ============================================================
# token_valuation.slurm —— SLURM 作业脚本（数组作业）
# 每个数组任务独立读取 CSV 的一部分行，做 token 估值。
# 提交方式（超算登录节点）：
#   cd /public/home/scno/sconqw05b/18-医疗AI模型系统
#   sbatch hpc/jobs/token_valuation.slurm
# ============================================================
#SBATCH --job-name=hpc_token_val
#SBATCH --partition=通用
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --array=0-15                   # 16 个子任务并行，可按需调整（0-31 / 0-63）
#SBATCH --output=/public/home/scno/sconqw05b/18-医疗AI模型系统/logs_hpc/slurm_%A_%a.out
#SBATCH --error=/public/home/scno/sconqw05b/18-医疗AI模型系统/logs_hpc/slurm_%A_%a.err

set -e
PROJECT_ROOT="/public/home/scno/sconqw05b/18-医疗AI模型系统"
cd "${PROJECT_ROOT}"

echo "[slurm] JOB_ID=${SLURM_JOB_ID} ARRAY_ID=${SLURM_ARRAY_TASK_ID} NODE=$(hostname)"
echo "[slurm] CPUS=${SLURM_CPUS_ON_NODE} MEM=${SLURM_MEM_PER_NODE}"
echo "[slurm] 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 激活虚拟环境
source "${PROJECT_ROOT}/env_py311/bin/activate"

# 运行 worker
python hpc/hpc_valuation_worker.py || exit 1

echo "[slurm] 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
