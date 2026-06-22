# ============================================================
# hpc_valuation_worker.py
# 在超算计算节点上运行 —— 按 SLURM_ARRAY_TASK_ID 分块读取 CSV
# 并把估值结果写入 outputs_hpc/ 下
# ============================================================
import os
import sys
import time
import math
import pandas as pd
import numpy as np
import yaml

from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.loader import HealthcareTokenLoader
from models.valuation_engine import AssetValuationEngine


# ---------- 读取配置 ----------
def load_config():
    cfg_path = os.path.join(ROOT, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------- 获取 SLURM 任务信息 ----------
def slurm_info():
    """返回 (task_id, total_tasks, total_nodes, node_id)"""
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    total_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    node_id = os.environ.get("SLURMD_NODENAME", "nid00000")
    total_nodes = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    return {
        "task_id": task_id,
        "total_tasks": total_tasks,
        "node_id": node_id,
        "total_nodes": total_nodes,
        "cpus_per_node": cpus_per_node,
    }


# ---------- 估值主逻辑 ----------
def run_valuation(cfg, slurm):
    dataset_path = cfg["data"]["dataset_path_hpc"] or cfg["data"]["dataset_path"]
    # 若超算端不存在主 CSV，回退到本地路径也不存在，则生成伪数据用于性能压测
    if not os.path.exists(dataset_path):
        alt = cfg["data"]["dataset_path"]
        if os.path.exists(alt):
            dataset_path = alt
    sample_per_task = int(cfg.get("hpc", {}).get("sample_per_task", 10_000_000))
    task_id = slurm["task_id"]
    total_tasks = slurm["total_tasks"]
    output_root = cfg["data"].get("output_dir_hpc") or cfg["data"].get("output_dir", os.path.join(ROOT, "outputs_hpc"))
    os.makedirs(output_root, exist_ok=True)

    # 若数据集不存在，则生成同分布的伪数据用于性能压测
    if not os.path.exists(dataset_path):
        print(f"[warn] 未找到数据集 {dataset_path}，将用伪数据做性能压测。")
        df = generate_synthetic_dataset(sample_per_task, task_id)
    else:
        # 计算总条数 + 每任务读取范围
        total_rows = _estimate_row_count(dataset_path)
        rows_per_task = math.ceil(total_rows / total_tasks)
        start_row = task_id * rows_per_task
        end_row = min(total_rows, start_row + rows_per_task)
        n_rows = end_row - start_row
        print(f"[info] task={task_id}/{total_tasks} 读取行: {start_row} -> {end_row} (共 {n_rows} 行)")
        df = pd.read_csv(
            dataset_path,
            skiprows=range(1, start_row + 1),  # 保留表头
            nrows=n_rows,
            encoding="utf-8",
        )

    # 估值引擎
    cat_counts = _category_counts(df, cfg)
    engine = AssetValuationEngine(config_path=os.path.join(ROOT, "config.yaml"))
    engine.update_category_counts(cat_counts)

    t0 = time.time()
    valued = engine.value_dataframe(df)
    dt = time.time() - t0
    n = len(valued)

    # 汇总
    eligible = valued[valued["eligible"] == True] if "eligible" in valued.columns else valued
    total_value = float(eligible["single_value"].sum())
    avg_value = float(eligible["single_value"].mean()) if len(eligible) else 0.0

    # 写 CSV（避免重复）
    out_csv = os.path.join(output_root, f"part_{task_id:06d}_of_{total_tasks:06d}.csv")
    # 只保留需要的列以减小体积
    keep_cols = [c for c in ["token_id","category","data_type","token_level",
                              "data_quality_score","compliance_score","single_value"] if c in valued.columns]
    valued[keep_cols].to_csv(out_csv, index=False, encoding="utf-8")

    # 写 JSON 摘要（便于上游汇总）
    summary = {
        "task_id": task_id,
        "total_tasks": total_tasks,
        "node_id": slurm["node_id"],
        "tokens_total": int(n),
        "tokens_eligible": int(len(eligible)),
        "total_value_cny": round(total_value, 2),
        "avg_value_per_token": round(avg_value, 2),
        "elapsed_sec": round(dt, 3),
        "tokens_per_sec": round(n / dt, 1) if dt > 0 else 0,
        "row_count_csv": total_rows if os.path.exists(dataset_path) else int(n),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    import json
    with open(os.path.join(output_root, f"part_{task_id:06d}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] task={task_id} 处理 {n} 条 | 耗时 {dt:.2f}s | 吞吐量 {summary['tokens_per_sec']} tok/s")
    print(f"       => {out_csv}")
    return summary


# ---------- 辅助工具 ----------
def _estimate_row_count(path):
    # 快速估算 CSV 总行数（不需要读全表）
    import subprocess
    try:
        # wc -l 最快，但 Windows 不支持；超算都是 Linux
        out = subprocess.check_output(["wc", "-l", path]).decode().strip()
        total = int(out.split()[0]) - 1  # 减去表头
        return max(total, 1)
    except Exception:
        # 回退：按文件大小粗略估算
        size_mb = os.path.getsize(path) / (1024 * 1024)
        # 本数据集约 593 MB = 100M 行 → 1 MB ≈ 168k 行
        return int(size_mb * 168_000)


def _category_counts(df: pd.DataFrame, cfg: dict) -> dict:
    if "category" in df.columns:
        vc = df["category"].value_counts().to_dict()
        return {str(k): int(v) for k, v in vc.items()}
    # 回退：按 config.yaml 中 categories 平均分布
    cats = list(cfg.get("categories", {}).keys())
    return {c: max(1, len(df) // len(cats)) for c in cats}


def generate_synthetic_dataset(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """当无真实数据时，生成同分布的伪数据用于性能压测"""
    rng = np.random.default_rng(seed + 42)
    categories = ["radiology","pathology","neurology","cardiology",
                  "laboratory","orthopedics","pediatrics","emergency"]
    data_types = ["image","text","ecg","lab","pathology","genetic","vital","other"]
    levels = ["A","B"]
    n = int(n_rows)
    df = pd.DataFrame({
        "token_id": [f"H-{seed}-{i:09d}" for i in range(n)],
        "domain": ["healthcare"] * n,
        "category": rng.choice(categories, size=n),
        "data_type": rng.choice(data_types, size=n),
        "entity_id": [f"ENT-{seed}-{rng.integers(0,999_999)}" for _ in range(n)],
        "data_quality_score": np.round(rng.uniform(94.0, 100.0, n), 2),
        "token_level": rng.choice(levels, size=n, p=[0.83, 0.17]),
        "completeness": np.round(rng.uniform(94.0, 100.0, n), 2),
        "accuracy": np.round(rng.uniform(94.0, 100.0, n), 2),
        "timeliness": np.round(rng.uniform(93.0, 99.0, n), 2),
        "compliance_score": np.round(rng.uniform(95.0, 100.0, n), 2),
        "created_at": [datetime(2026,1,1).strftime("%Y-%m-%d %H:%M:%S")] * n,
    })
    return df


# ---------- 主入口 ----------
if __name__ == "__main__":
    cfg = load_config()
    slurm = slurm_info()
    print(f"[start] task_id={slurm['task_id']} total_tasks={slurm['total_tasks']} "
          f"node={slurm['node_id']} cpus/node={slurm['cpus_per_node']}")
    summary = run_valuation(cfg, slurm)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[end] worker 退出")
