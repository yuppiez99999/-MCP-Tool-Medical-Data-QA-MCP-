# ============================================================
# aggregate_hpc_reports.py —— 汇总超算并行作业的结果
# 遍历 outputs_hpc/part_*_summary.json，汇总生成 Markdown 报告
# 使用（超算登录节点）:
#   python hpc/aggregate_hpc_reports.py --dir outputs_hpc
# ============================================================
import argparse
import glob
import json
import os
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs_hpc", help="part 汇总 JSON 所在目录")
    p.add_argument("--out", default="outputs/hpc_aggregate_report.md")
    args = p.parse_args()

    in_dir = os.path.abspath(args.dir)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "part_*_summary.json")))
    if not files:
        print(f"[warn] 未在 {in_dir} 中找到任何 part_*_summary.json")
        return

    summaries = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                summaries.append(json.load(fp))
        except Exception as e:
            print(f"[warn] 跳过 {os.path.basename(f)}: {e}")

    total_tokens = sum(s["tokens_total"] for s in summaries)
    total_eligible = sum(s["tokens_eligible"] for s in summaries)
    total_value = sum(s["total_value_cny"] for s in summaries)
    elapsed_min = max([s["elapsed_sec"] for s in summaries]) / 60.0
    avg_tps = total_tokens / max(1.0, sum(s["elapsed_sec"] for s in summaries))

    nodes = sorted({s.get("node_id", "?") for s in summaries})

    lines = []
    lines.append("# 超算中心 · 医疗健康 Token 估值汇总报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 汇总任务数: **{len(summaries)}**")
    lines.append(f"- 参与节点数: **{len(nodes)}** ({', '.join(nodes[:10])}{' ...' if len(nodes)>10 else ''})")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 处理 Token 总数 | {total_tokens:,} 条 |")
    lines.append(f"| 符合登记条件 Token 数 | {total_eligible:,} 条 |")
    lines.append(f"| 资产估值合计 | ¥{total_value:,.0f} |")
    lines.append(f"| 单条均值价值 | ¥{(total_value/total_eligible if total_eligible else 0):,.2f} |")
    lines.append(f"| 集群总吞吐量 | {avg_tps:,.0f} tok/s |")
    lines.append(f"| 最长任务耗时 | {elapsed_min:.1f} 分钟 |")
    lines.append("")
    lines.append("## 子任务明细")
    lines.append("")
    lines.append("| task_id | 节点 | 处理条数 | 吞吐量(tok/s) | 资产价值(¥) | 耗时(s) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s.get('task_id','?')} | {s.get('node_id','?')} | "
            f"{s.get('tokens_total',0):,} | {s.get('tokens_per_sec',0):,.0f} | "
            f"{s.get('total_value_cny',0):,.0f} | {s.get('elapsed_sec',0):.1f} |"
        )
    lines.append("")
    lines.append("## 登记建议")
    lines.append("")
    lines.append("> 建议将符合条件的 **{:,}** 条 Token（价值 ¥{:,.0f}）在北数所分批登记，每批 10M 条，"
                 "以充分利用数据资产价值。".format(total_eligible, total_value))
    lines.append("")
    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n[done] 报告已保存到: {out_path}")


if __name__ == "__main__":
    main()
