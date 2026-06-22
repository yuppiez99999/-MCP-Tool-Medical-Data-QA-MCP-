# 医疗AI模型系统 · 超算中心部署与 Token 生成能力估算

> 生成日期：2026-06-16
> 目标：把 [18-医疗AI模型系统](file:///e:/各种PY程序/18-医疗AI模型系统) 项目部署到超算中心，
> 批量完成 100M 条 Token 的估值登记，并在本地 PC 先估算「在超算中心可以多生成多少 Token」。

---

## 一、文件总览

所有超算相关脚本与配置集中在以下目录，便于打包与同步：

```
18-医疗AI模型系统/
├── README.md                                           # 全局说明 + 第七章「超算部署与估算」
├── config.yaml                                         # 新增 hpc 段、dataset_path_hpc、output_dir_hpc
├── hpc/
│   ├── deploy_config.sh                                # 超算账号 / 路径 / 节点规格（按需改）
│   ├── deploy_main.sh                                  # 超算登录节点一键部署（创建 env + 提交 SLURM）
│   ├── build_and_upload.ps1                            # Windows 本地：打包 + scp + 超算端部署
│   ├── estimate_tokens_on_hpc.py                       # 本地估算脚本（已验证可运行 ✅）
│   ├── hpc_valuation_worker.py                         # 超算计算节点 worker（读 CSV → 估值 → 写 outputs_hpc/）
│   ├── aggregate_hpc_reports.py                        # 汇总 outputs_hpc/part_*_summary.json → Markdown
│   └── jobs/
│       └── token_valuation.slurm                       # SLURM 数组作业（默认 16 路并行）
└── outputs/
    └── hpc_estimate_report.md                          # estimate_tokens_on_hpc.py 自动生成的估算报告
```

关键文件直接链接（点击打开）：

- [deploy_config.sh](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/deploy_config.sh)
- [deploy_main.sh](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/deploy_main.sh)
- [build_and_upload.ps1](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/build_and_upload.ps1)
- [estimate_tokens_on_hpc.py](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/estimate_tokens_on_hpc.py)
- [hpc_valuation_worker.py](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/hpc_valuation_worker.py)
- [aggregate_hpc_reports.py](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/aggregate_hpc_reports.py)
- [token_valuation.slurm](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/jobs/token_valuation.slurm)
- [config.yaml](file:///e:/各种PY程序/18-医疗AI模型系统/config.yaml)
- [hpc_estimate_report.md](file:///e:/各种PY程序/18-医疗AI模型系统/outputs/hpc_estimate_report.md)

---

## 二、超算账号与路径配置（先改好这里）

编辑 [hpc/deploy_config.sh](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/deploy_config.sh)，至少核对以下字段：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `HPC_USER` | `sconqw05b` | 超算中心的登录用户名 |
| `HPC_HOST` | `scno.hpccube.com` | 超算中心登录节点主机名 / IP |
| `HPC_SSH_KEY` | `~/ssh/scno_key` | SSH 私钥路径（需上传公钥到超算） |
| `HPC_PROJECT_ROOT` | `/public/home/scno/sconqw05b/18-医疗AI模型系统` | 项目在超算端的根目录 |
| `HPC_DATASET` | `$HPC_PROJECT_ROOT/healthcare_token_A_B_100M.csv` | 医疗 Token 数据集在超算端路径 |
| `HPC_PARTITION` | `通用` | 超算分区（通用/GPU/debug 等，按实际改） |
| `HPC_NODES` | `4` | 申请节点数 |
| `HPC_CPUS_PER_TASK` | `64` | 每节点 CPU 核数 |
| `HPC_TIME_LIMIT` | `08:00:00` | SLURM 作业最长运行时长 |

同步地，[config.yaml](file:///e:/各种PY程序/18-医疗AI模型系统/config.yaml)
中新增的 `hpc` 段与超算端路径，在超算端 worker 里会自动读取：

```yaml
data:
  dataset_path_hpc: "/public/home/scno/sconqw05b/18-医疗AI模型系统/healthcare_token_A_B_100M.csv"
  output_dir_hpc:    "/public/home/scno/sconqw05b/18-医疗AI模型系统/outputs_hpc"

hpc:
  sample_per_task: 10000000
  array_tasks: 16
  cpus_per_node: 64
  mem_per_node_gb: 110
  partition: "通用"
```

---

## 三、三种部署方式（三选一）

### 方式 A · 超算登录节点手工部署（**第一次推荐**，便于调试）

1. 使用 SCP / rsync / Git 把整个 `18-医疗AI模型系统` 目录上传到登录节点：

   ```bash
   # Windows PowerShell
   scp -i ~\.ssh\scno_key -r "e:\各种PY程序\18-医疗AI模型系统" sconqw05b@scno.hpccube.com:/public/home/scno/sconqw05b/
   # Linux / macOS
   scp -i ~/.ssh/scno_key -r 18-医疗AI模型系统 sconqw05b@scno.hpccube.com:/public/home/scno/sconqw05b/
   ```

2. 在登录节点创建虚拟环境并安装依赖：

   ```bash
   cd /public/home/scno/sconqw05b/18-医疗AI模型系统
   python3 -m venv env_py311
   source env_py311/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 提交 SLURM 数组作业（默认 16 路并行，每路读 CSV 的一块行）：

   ```bash
   sbatch hpc/jobs/token_valuation.slurm
   squeue -u sconqw05b                 # 查看作业状态（PD 排队 / R 运行中）
   ```

4. 作业完成后，汇总并生成登记报告：

   ```bash
   source env_py311/bin/activate
   python hpc/aggregate_hpc_reports.py --dir outputs_hpc --out outputs/hpc_aggregate_report.md
   # 同时可运行登记脚本生成 JSON
   python scripts/generate_registration_report.py --sample 100000000 --domain healthcare
   ```

### 方式 B · 超算登录节点一键部署

把项目上传好后，在登录节点直接执行：

```bash
cd /public/home/scno/sconqw05b/18-医疗AI模型系统
bash hpc/deploy_main.sh
```

脚本会自动：
1. 检查与登录节点的 SSH 连接；
2. 创建项目目录并上传（如果在登录节点运行本脚本时，tar 包需要已事先上传）；
3. 创建 `env_py311` 虚拟环境并安装依赖；
4. 调用 `sbatch hpc/jobs/token_valuation.slurm` 提交作业并打印 `JOB_ID`。

### 方式 C · Windows 本地 PowerShell 一键打包+上传+提交

在本地 Windows `e:\各种PY程序` 下执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\18-医疗AI模型系统\hpc\build_and_upload.ps1
```

脚本自动完成以下步骤：
1. **本地**：tar.gz 打包整个 `18-医疗AI模型系统`（排除 `outputs / __pycache__ / *.log`）
2. **本地 → 超算**：用 `scp` 把 tar.gz 上传到 `sconqw05b@scno.hpccube.com:$HPC_PROJECT_ROOT`
3. **超算登录节点**：自动 `tar -xzf` 解压 → 创建 `env_py311` → `pip install -r requirements.txt` → `sbatch hpc/jobs/token_valuation.slurm`
4. 输出 `SLURM_JOB_ID`，并提示用 `squeue -u sconqw05b` 查看队列

> ⚠️ 首次在 Windows 运行前，务必确认：
> - 本机有 OpenSSH 客户端（`ssh / scp` 可用；Windows 10+ 默认自带）；
> - 已把本地公钥 `~/.ssh/scno_key.pub` 添加到超算登录节点的 `~/.ssh/authorized_keys`；
> - 能无密码登录：`ssh -i ~\.ssh\scno_key -o StrictHostKeyChecking=no sconqw05b@scno.hpccube.com`。

---

## 四、超算中心作业流程（内部流程，便于排错）

```
┌──────────────────────────────────────────────────────────────┐
│  1. 登录节点 (login node)                                      │
│     sbatch hpc/jobs/token_valuation.slurm                    │
│     └─ 向 "通用" 分区提交 16 个任务（SLURM_ARRAY_TASK_ID=0..15）│
│                                                                │
│  2. 每个计算节点 (compute node) 独立执行：                      │
│     python hpc/hpc_valuation_worker.py                        │
│       ├─ 读取 SLURM_ARRAY_TASK_ID，定位 CSV 行范围              │
│       ├─ 读 healthcare_token_A_B_100M.csv（跳过其余块）         │
│       ├─ 通过 AssetValuationEngine 做批量估值                  │
│       ├─ 写 outputs_hpc/part_XXXXXX.csv + part_XXXXXX_summary.json │
│       └─ 打印吞吐量、合格条数、资产价值合计                     │
│                                                                │
│  3. 所有作业完成后，在登录节点汇总：                            │
│     python hpc/aggregate_hpc_reports.py --dir outputs_hpc     │
│       └─ 生成 outputs/hpc_aggregate_report.md                 │
└──────────────────────────────────────────────────────────────┘
```

[hpc_valuation_worker.py](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/hpc_valuation_worker.py)
的关键设计点：

- 只根据 `SLURM_ARRAY_TASK_ID` 读取一段 CSV（通过 `skiprows + nrows`），不会把整份 593 MB CSV 载入内存；
- 若数据集不存在，自动按同分布生成伪数据，便于性能压测（不影响正式登记）；
- 每个任务写出独立的 `part_XXXXXX.csv / part_XXXXXX_summary.json`，避免多任务写入冲突；
- 汇总脚本 `aggregate_hpc_reports.py` 读 JSON 汇总，输出结构化 Markdown，方便直接用于北数所登记。

[token_valuation.slurm](file:///e:/各种PY程序/18-医疗AI模型系统/hpc/jobs/token_valuation.slurm)
的关键参数（按需改）：

| 项 | 值 | 说明 |
|---|---|---|
| `--nodes` | 1 | 每个数组任务占 1 个节点 |
| `--ntasks-per-node` | 1 | 每节点 1 个主进程，脚本内用 pandas 多线程估值 |
| `--cpus-per-task` | 64 | 每节点 CPU 核数（按需修改） |
| `--mem` | 110G | 每节点内存上限 |
| `--time` | 08:00:00 | 最长运行时长 |
| `--array` | 0-15 | 16 路并行；改 `0-31` 可扩展到 32 路 |

---

## 五、在本地 PC 估算「超算中心可以多生成多少 Token」

### 5.1 运行方式

```bash
python e:\各种PY程序\18-医疗AI模型系统\hpc\estimate_tokens_on_hpc.py
```

脚本同时会把报告写到
[outputs/hpc_estimate_report.md](file:///e:/各种PY程序/18-医疗AI模型系统/outputs/hpc_estimate_report.md)。

### 5.2 假设基准（可在脚本顶部修改）

| 项目 | 默认值 | 说明 |
|---|---|---|
| 本地 PC | 8 核 @ 3.5 GHz / 32 GB | 用于基线对比 |
| Token 吞吐量基线 | 1,250 tok/s | 由前序作业实测（100k 条 / 80 秒） |
| 单条 Token 均值 | ¥57.56 | 由 100M 条估值汇总得出 |
| 1 核时成本 | ≈ ¥1.0 | 超算 CPU 计费参考 |
| 1 GPU 时成本 | ≈ ¥12.0 | 超算 GPU 计费参考 |
| 目标数据集 | 100,000,000 条 / 593 MB | 医疗健康 Token 数据集 |

### 5.3 各硬件配置的吞吐量与可生成 Token 数

| 场景 | 节点数 | 吞吐量 (tok/s) | 8 小时总 Token 数 | 相对本地 PC 加速比 |
|---|---:|---:|---:|---:|
| 本地 PC（8 核） | 1 | 1,250 | ~36,000,000 | ×1.0 |
| 单节点 CPU（通用区） | 1 | 8,500 | ~244,800,000 | ×6.8 |
| **4 节点 CPU（推荐日常批处理）** | 4 | **34,000** | **~979,200,000** | **×27.2** |
| 16 节点 CPU | 16 | 136,000 | ~3,916,800,000 | ×108.8 |
| 1 GPU 节点 | 1 | 52,275 | ~1,502,640,000 | ×41.8 |
| 4 GPU 节点 | 4 | 209,100 | ~6,022,080,000 | ×167.3 |
| **8 GPU 节点 × 24h 天级批处理** | 8 | **418,200** | **~36,132,480,000** | **×334.6** |

### 5.4 处理「目标 100M 条 Token 数据集」所需时长与成本

| 配置 | 完成时长 | 估算成本 (¥) | 适用场景 |
|---|---:|---:|---|
| 本地 PC | 22.2 小时 | ≈ 11（电费） | 小样本调试 |
| 4 节点 CPU | 2.4 小时 | ≈ 602 | **日常登记批处理** |
| 16 节点 CPU | 0.6 小时 | ≈ 602 | 大规模 / 赶截止日期 |
| 8 GPU 节点 | 0.2 小时 | ≈ 18 | 训练模型 / 向量打分加速 |

### 5.5 数据资产规模（100M 条 Token 计）

- 总 Token 数：**100,000,000** 条
- 平均单条价值：**¥57.56**
- 整包资产估值：**¥5,756,000,000**
- 北数所登记建议（企业批量 85 折）：**¥4,892,600,000**

---

## 六、快速排错 Checklist

| 现象 | 可能原因 | 解决 |
|---|---|---|
| `ssh: connect to host scno.hpccube.com port 22: Connection timed out` | 未在超算白名单 / VPN 中运行 | 先连单位 VPN 或白名单跳板机再 SSH |
| `sbatch: error: Batch job submission failed: Invalid partition name specified` | 分区名不对，例如「通用」需对应超算实际分区 | 登录节点执行 `sinfo` 查询可用分区，修改 `--partition=` 或 `deploy_config.sh` 中的 `HPC_PARTITION` |
| 作业一直在排队 (`ST=PD`) | 资源申请过大或队列紧张 | 把 `--array=0-15` 缩小到 `0-7`，把 `--mem` 降为 `64G`，把 `--cpus-per-task` 降为 `32` |
| 超算端 Python 报 `No module named 'pandas'` | 忘记激活 `env_py311` 或 `requirements.txt` 没装全 | 在超算端执行 `source env_py311/bin/activate && pip install -r requirements.txt` |
| `hpc_valuation_worker.py` 运行正常但 `outputs_hpc/` 为空 | 工作路径错误导致 worker 输出到了其它目录 | 在 SLURM 脚本里加 `cd /public/home/scno/sconqw05b/18-医疗AI模型系统` 并确认 `output_dir_hpc` 配置正确 |
| `wc -l` 不可用（非 Linux 环境） | worker 会退回到「文件大小粗估」模式 | 只影响总行数估算，不影响估值质量；建议仍在 Linux 超算节点运行 |
| 作业崩溃 OOM (`Out Of Memory`) | 单块 CSV 超过节点内存 | 把 `config.yaml` 的 `hpc.sample_per_task` 改小，或加大 `--array` 并行任务数 |

---

## 七、后续建议

1. **先小规模跑一次**：把 `--array=0-1` 或把 `sample_per_task` 改到 `2,000,000`，跑通一条完整链路（提交 → worker 写 outputs_hpc → aggregate_hpc_reports 汇总）。
2. **登记用 4 节点 CPU × 8 小时**：100M 条 Token 可在 ~2 小时内处理完，成本约 ¥600；足够作为北数所登记批次的输出。
3. **月/季度批处理**：建议 `16 节点 CPU × 8 小时` 一次跑完 ~40 亿条，或 `8 GPU 节点 × 24h` 做训练+估值的天级批处理；把流程固化到 `deploy_main.sh` 后每次只需一行命令。
4. **与现有 FastAPI 结合**：超算端生成的 `outputs_hpc/*.csv / *.json` 可直接作为 `api/healthcare_ai_extension.py` 的数据来源（把 `data.dataset_path_hpc` 指向汇总后的 CSV 即可），前端 / 北数所登记流程完全复用现有 `/api/healthcare/asset-summary / generate-registration` 等端点。
5. **监控与告警**：登录节点可放一条 `crontab -e` 的定时检查脚本，在 outputs_hpc 下所有 `part_*_summary.json` 写齐后自动触发 `aggregate_hpc_reports.py` 并邮件/企业微信通知。
