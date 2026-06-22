#!/usr/bin/env bash
# ============================================================
# deploy_config.sh —— 医疗AI模型系统超算部署配置
# 使用方法（Windows PowerShell / Linux Bash 均适用）:
#   Windows: .\18-医疗AI模型系统\hpc\build_and_upload.ps1
#   Linux / 超算登录节点:
#       cd /public/home/scno/sconqw05b/18-医疗AI模型系统
#       bash hpc/deploy_main.sh
# ============================================================

# ---------- 超算账户 ----------
export HPC_USER="sconqw05b"
export HPC_HOST="scno.hpccube.com"
export HPC_PORT="22"
export HPC_SSH_KEY="${HOME}/.ssh/id_rsa_scnet"

# ---------- 项目路径 ----------
export HPC_PROJECT_ROOT="/public/home/scno/sconqw05b/18-医疗AI模型系统"
export HPC_OUTPUT_ROOT="${HPC_PROJECT_ROOT}/outputs_hpc"
export HPC_LOG_ROOT="${HPC_PROJECT_ROOT}/logs_hpc"
export HPC_ENV_ROOT="${HPC_PROJECT_ROOT}/env_py311"

# ---------- 数据集 ----------
# 主数据集 CSV 放于 01数据文件/healthcare_token_A_B_100M.csv
# 若仅做全量估值，100M 即可；若做全量生成，需要更大原始数据
export HPC_DATASET="${HPC_PROJECT_ROOT}/healthcare_token_A_B_100M.csv"
export HPC_DATASET_SIZE_MB="593"          # 约 593 MB（100M 样本）

# ---------- 计算节点配置（按你截图的分区调整）----------
# 通用区：64 核 CPU、128G 内存
# GPU 区：NVIDIA A100 / H800 48G / 80G
export HPC_PARTITION="通用"                # 按实际修改：通用 / GPU / debug
export HPC_NODES=4                         # 节点数
export HPC_TASKS_PER_NODE=1                # 每节点 1 个主任务，脚本内并行
export HPC_CPUS_PER_TASK=64                # 每个任务 CPU 核数
export HPC_MEM_PER_NODE="110G"             # 每节点内存
export HPC_GPU_PER_NODE=0                  # 纯 CPU 估值跑 0；训练模型改为 1-8
export HPC_TIME_LIMIT="08:00:00"           # SLURM 时间上限（8 小时）

# ---------- 业务参数 ----------
export HPC_SAMPLE_PER_TASK="10000000"       # 每个子任务处理多少条 token（默认1千万）
export HPC_OUTPUT_FILENAME_PREFIX="healthcare_token_A_B"
export HPC_QUALITY_MIN="94"                 # 过滤：quality >= 94
export HPC_COMPLIANCE_MIN="95"              # 合规分阈值
export HPC_MIN_VALUE_PER_TOKEN="15"         # 单条估值下限（元）
export HPC_MAX_VALUE_PER_TOKEN="150"        # 单条估值上限

echo "[config] 已加载医疗AI超算配置 -> ${HPC_PROJECT_ROOT}"
echo "[config] 分区=${HPC_PARTITION} 节点=${HPC_NODES} 每节点CPU=${HPC_CPUS_PER_TASK}"
