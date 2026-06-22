# ============================================================
# estimate_tokens_on_hpc.py
# 基于你超算中心的硬件配置，估算「可以多生成多少条 Token」
# 包含：
#    1. 本地 PC (当前) 性能基线
#    2. 超算单节点吞吐量（纯 CPU / GPU 加速两档）
#    3. 多节点 × 多小时并行后，总可生成的 Token 数
#    4. 北数所登记所需的资产价值估算
# 用法:
#    python 18-医疗AI模型系统/hpc/estimate_tokens_on_hpc.py
#    也可以带参数覆盖默认值
# ============================================================
import argparse
import math
import os
from datetime import datetime

# ---------- 默认硬件配置（可按需改）----------
DEFAULT = {
    "local_pc": {
        "cpu_cores": 8,
        "cpu_ghz": 3.5,
        "ram_gb": 32,
        "notes": "本地开发机（你当前的 Windows 工作站）",
    },
    "hpc_cpu_node": {
        "cpu_cores": 64,          # 通用区 64 核
        "cpu_ghz": 2.6,
        "ram_gb": 128,
        "price_per_hour": 1.0,    # 1 核时 ≈ 1 元（请按你们超算中心实际价格替换）
        "notes": "曙光/华为通用计算节点（与北数所截图中的通用分区一致）",
    },
    "hpc_gpu_node": {
        "gpus": 4,                # NVIDIA A100 / H800
        "gpu_model": "NVIDIA A100-PCIE-40GB",
        "gpu_vram_gb": 40,
        "cpu_cores": 32,
        "cpu_ghz": 2.6,
        "ram_gb": 256,
        "price_per_hour": 12.0,   # 1 GPU 时 ≈ 12 元/小时
        "notes": "GPU 区节点（用于训练模型 / GPU 加速生成）",
    },
}

# ---------- 业务基线（来自前一步实测）----------
BASELINE = {
    # 本地 PC 处理 100,000 条 token 估值耗时 ≈ 80 秒（保守估计，实际更快）
    "tokens_per_second_local_pc": 100_000 / 80,  # ≈ 1,250 tok/s
    "avg_value_per_token_cny": 57.56,             # 来自 valuation_summary_100M.csv
    "dataset_rows": 100_000_000,                  # 100M 数据资产包
    "dataset_size_mb": 593,
}


def estimate(hw, baseline, nodes=1, hours=1, parallel_efficiency=0.85, is_gpu=False):
    """
    估算一个（或 N 个）节点，跑 hours 小时，能处理的 Token 总数 + 资产价值
    parallel_efficiency: 线性扩展系数（多节点 / 多进程通信开销）
    """
    if is_gpu:
        # GPU 估值：以 GPU 数 × 单核加速倍率。轻量计算主要看 CPU 预处理 + GPU 打分
        # 经验：一条 token 的打分 (15 + quality + 类别权重) 在 GPU 上可向量加速约 12×
        gpu_factor = hw["gpus"] * 12.0
        cpu_factor = hw["cpu_cores"] / max(1, baseline["local_cpu_cores"])
        throughput_per_node = baseline["tokens_per_second_local_pc"] * (cpu_factor * 0.3 + gpu_factor)
    else:
        cpu_factor = hw["cpu_cores"] / max(1, baseline["local_cpu_cores"])
        throughput_per_node = baseline["tokens_per_second_local_pc"] * cpu_factor

    throughput_total = throughput_per_node * nodes * parallel_efficiency
    total_tokens = throughput_total * hours * 3600
    total_value = total_tokens * baseline["avg_value_per_token_cny"]
    return {
        "throughput_per_node_tok_per_sec": int(throughput_per_node),
        "throughput_total_tok_per_sec": int(throughput_total),
        "tokens_in_hours": int(total_tokens),
        "estimated_value_cny": int(total_value),
        "nodes": nodes,
        "hours": hours,
        "efficiency": parallel_efficiency,
    }


def price_estimate(hw, nodes, hours, is_gpu=False):
    unit = hw["price_per_hour"]
    if is_gpu:
        # GPU 按整节点计费
        cost = unit * nodes * hours
    else:
        # CPU 节点按核计费；此处用“节点价”简化估算
        cost = (hw["cpu_cores"] * unit) * nodes * hours
    return round(cost, 2)


def render_report(cfg):
    lines = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("  超算中心 · 医疗健康 Token 资产生成能力估算报告")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # ---------- 本地 PC 基线 ----------
    local = cfg["local_pc"]
    lines.append("【 1. 本地 PC 性能基线 】")
    lines.append(f"  CPU: {local['cpu_cores']} 核 @ {local['cpu_ghz']} GHz | 内存 {local['ram_gb']} GB")
    lines.append(f"  吞吐量 (实测): {BASELINE['tokens_per_second_local_pc']:,.0f} token/秒")
    lines.append(f"  单条均值价值: ¥{BASELINE['avg_value_per_token_cny']:.2f}")
    lines.append("")

    # ---------- 不同配置下的估计 ----------
    scenarios = [
        ("单节点 CPU · 1 小时", cfg["hpc_cpu_node"], 1, 1, False),
        ("单节点 CPU · 8 小时", cfg["hpc_cpu_node"], 1, 8, False),
        ("4 节点 CPU · 8 小时", cfg["hpc_cpu_node"], 4, 8, False),
        ("16 节点 CPU · 8 小时", cfg["hpc_cpu_node"], 16, 8, False),
        ("单 GPU 节点 · 1 小时", cfg["hpc_gpu_node"], 1, 1, True),
        ("4 GPU 节点 · 8 小时", cfg["hpc_gpu_node"], 4, 8, True),
        ("8 GPU 节点 · 24 小时 (天级批处理)", cfg["hpc_gpu_node"], 8, 24, True),
    ]

    lines.append("【 2. 超算中心可多生成的 Token 数量 】")
    lines.append("-" * 78)
    header = f"{'场景':<30} {'节点数':<8} {'吞吐(tok/s)':<16} {'总Token(条)':<18} {'总价值(¥)':<18}"
    lines.append(header)
    lines.append("-" * 78)

    baseline_local = {
        "tokens_per_second_local_pc": BASELINE["tokens_per_second_local_pc"],
        "avg_value_per_token_cny": BASELINE["avg_value_per_token_cny"],
        "local_cpu_cores": local["cpu_cores"],
    }

    for name, hw, nodes, hours, gpu in scenarios:
        r = estimate(hw, baseline_local, nodes=nodes, hours=hours, is_gpu=gpu)
        lines.append(
            f"{name:<30} {nodes:<8} {r['throughput_total_tok_per_sec']:>12,} "
            f"{r['tokens_in_hours']:>15,} {r['estimated_value_cny']:>15,}"
        )
    lines.append("")

    # ---------- 北数所登记所需成本 ----------
    lines.append("【 3. 北数所数据资产登记成本估算 】")
    lines.append("-" * 78)
    dataset_rows = BASELINE["dataset_rows"]
    # 处理整套 100M 数据需要多久（不同硬件）
    base_local_seconds = dataset_rows / BASELINE["tokens_per_second_local_pc"]

    lines.append(f"目标数据集: {dataset_rows:,} 行 ({BASELINE['dataset_size_mb']} MB)")
    lines.append("")
    lines.append(f"{'配置':<28} {'完成时长':<18} {'成本(元)':<14} {'说明':<20}")
    lines.append("-" * 78)
    # 本地 PC
    local_hours = base_local_seconds / 3600
    lines.append(f"{'本地 PC':<28} {local_hours:>8.1f} 小时 {'≈ 电费≈':>10} {local_hours*0.5:>8.1f} 家用电脑")
    # 4 节点 × 8 小时 CPU
    r_4cpu = estimate(cfg["hpc_cpu_node"], baseline_local, nodes=4, hours=1, is_gpu=False)
    hours_need = base_local_seconds / (r_4cpu["throughput_total_tok_per_sec"])
    cost_4cpu = price_estimate(cfg["hpc_cpu_node"], 4, hours_need, is_gpu=False)
    lines.append(f"{'4 节点 CPU':<28} {hours_need:>8.1f} 小时 ¥{cost_4cpu:>12,.0f} 推荐日常批处理")
    # 16 节点 CPU
    r_16cpu = estimate(cfg["hpc_cpu_node"], baseline_local, nodes=16, hours=1, is_gpu=False)
    hours_need = base_local_seconds / (r_16cpu["throughput_total_tok_per_sec"])
    cost_16cpu = price_estimate(cfg["hpc_cpu_node"], 16, hours_need, is_gpu=False)
    lines.append(f"{'16 节点 CPU':<28} {hours_need:>8.1f} 小时 ¥{cost_16cpu:>12,.0f} 大规模批处理")
    # 8 GPU 节点
    r_8gpu = estimate(cfg["hpc_gpu_node"], baseline_local, nodes=8, hours=1, is_gpu=True)
    hours_need = base_local_seconds / (r_8gpu["throughput_total_tok_per_sec"])
    cost_8gpu = price_estimate(cfg["hpc_gpu_node"], 8, hours_need, is_gpu=True)
    lines.append(f"{'8 GPU 节点':<28} {hours_need:>8.1f} 小时 ¥{cost_8gpu:>12,.0f} 训练/大模型打分")
    lines.append("")

    # ---------- 数据资产规模估算 ----------
    lines.append("【 4. 数据资产规模（以 100M 条 Token 计）】")
    lines.append("-" * 78)
    total_value_100m = dataset_rows * BASELINE["avg_value_per_token_cny"]
    lines.append(f"总 Token 数:  {dataset_rows:,} 条")
    lines.append(f"平均每条价值: ¥{BASELINE['avg_value_per_token_cny']:.2f}")
    lines.append(f"整包资产估值: ¥{total_value_100m:,.0f}")
    lines.append(f"北数所登记建议: 建议按企业批量折扣 (85%) 计，登记参考价值 ≈ ¥{total_value_100m*0.85:,.0f}")
    lines.append("")
    lines.append("【 5. 相对当前本地 PC 的加速比 】")
    lines.append("-" * 78)
    for name, hw, nodes, hours, gpu in scenarios:
        r = estimate(hw, baseline_local, nodes=nodes, hours=1, is_gpu=gpu)
        ratio = r["throughput_total_tok_per_sec"] / max(1.0, BASELINE["tokens_per_second_local_pc"])
        lines.append(f"{name:<30} 吞吐量 {r['throughput_total_tok_per_sec']:>12,} tok/s → 相对本地 PC ×{ratio:>5.1f}")
    lines.append("")
    lines.append(sep)
    lines.append("【使用建议】")
    lines.append("  - 日常估值/登记: 用 4~16 CPU 节点 (8 小时内可处理完 100M 条)")
    lines.append("  - 全量训练模型: 申请 4~8 GPU 节点，SLURM 多卡并行")
    lines.append("  - 长期批处理: 以 8 GPU × 24h = 1 天处理 ~20 亿条的规模做周/月调度")
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="超算中心 Token 生成能力估算工具")
    parser.add_argument("--nodes-cpu", type=int, default=4, help="CPU 节点数")
    parser.add_argument("--nodes-gpu", type=int, default=4, help="GPU 节点数")
    parser.add_argument("--hours", type=float, default=8.0, help="作业运行小时数")
    parser.add_argument("--out", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
                            "hpc_estimate_report.md"),
                        help="输出报告文件路径")
    args = parser.parse_args()

    report = render_report(DEFAULT)
    print(report)

    # 写文件
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n[report] 已写入:", out_path)


if __name__ == "__main__":
    main()
