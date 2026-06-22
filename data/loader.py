# -*- coding: utf-8 -*-
"""
data / loader.py
真实数据集加载器 — 5,000,000 条 A级/B级 医疗 Token

策略:
1. 首次启动: 分块读取 CSV → 分层采样 50,000 条 → 缓存为 Parquet
2. 后续启动: 直接加载 Parquet (毫秒级)
3. 提供统计信息缓存 (按科室/等级/数据类型聚合)
"""
import os
import hashlib
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np


# ============================================================
# 常量
# ============================================================
# 数据集路径（请通过环境变量或config.yaml配置，此处为兜底默认值）
DATASET_PATH = os.environ.get(
    "MEDICAL_DATASET_PATH",
    "./data/healthcare_token_dataset.csv"
)
SAMPLE_PATH = os.environ.get(
    "MEDICAL_SAMPLE_PATH",
    "./outputs/data_sample.parquet"
)
STATS_PATH = os.environ.get(
    "MEDICAL_STATS_PATH",
    "./outputs/data_stats.json"
)
SAMPLE_SIZE = 50000  # 分层采样规模
CHUNK_SIZE = 100000  # 分块读取


# 真实数据类型 → 科室映射 (基于数据集实际分布)
TYPE_DEPT_MAP = {
    "ct_image":        "radiology",     # CT影像 → 放射科
    "blood_test":      "laboratory",    # 血液检验 → 检验科
    "pathology_slide": "pathology",     # 病理切片 → 病理科
    "ecg":             "cardiology",    # 心电图 → 心血管科
    "ultrasound":      "radiology",     # 超声 → 放射科 (实际数据中归放射科)
    "x_ray":           "orthopedics",   # X光 → 骨科
    "growth_record":   "pediatrics",    # 生长记录 → 儿科
    "triage":          "emergency",     # 分诊 → 急诊科
}

# 数据类型中文名
DATA_TYPE_CN = {
    "ct_image": "CT影像", "blood_test": "血液检验",
    "pathology_slide": "病理切片", "ecg": "心电图",
    "ultrasound": "超声", "x_ray": "X光",
    "growth_record": "生长记录", "triage": "分诊记录",
}


# ============================================================
# 数据加载器
# ============================================================
class DataLoader:
    """真实数据集加载器（懒加载 + Parquet 缓存）"""

    _instance: Optional["DataLoader"] = None
    _sample_df: Optional[pd.DataFrame] = None
    _stats: Optional[Dict[str, Any]] = None

    @classmethod
    def instance(cls) -> "DataLoader":
        """单例模式"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        os.makedirs(os.path.dirname(SAMPLE_PATH), exist_ok=True)

    # ---------------------------
    # 加载采样数据
    # ---------------------------
    def load_sample(self, force_rebuild: bool = False) -> pd.DataFrame:
        """加载采样数据 (50,000条分层采样)

        Args:
            force_rebuild: 强制重新生成采样文件
        """
        if self._sample_df is not None and not force_rebuild:
            return self._sample_df

        if os.path.exists(SAMPLE_PATH) and not force_rebuild:
            self._sample_df = pd.read_parquet(SAMPLE_PATH)
            return self._sample_df

        # 分层采样: 按科室+等级分组
        print(f"[DataLoader] 首次启动，开始分层采样 {SAMPLE_SIZE} 条...")
        chunks = pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE)
        all_chunks = []
        for i, chunk in enumerate(chunks):
            # 每块按 1/100 比例采样
            sampled = chunk.groupby(["category", "token_level"], group_keys=False).apply(
                lambda g: g.sample(frac=0.01, random_state=42)
            )
            all_chunks.append(sampled)
            if (i + 1) % 10 == 0:
                print(f"  已处理 {(i + 1) * CHUNK_SIZE:,} 行")

        df = pd.concat(all_chunks, ignore_index=True)
        # 调整到目标采样数
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        # 添加派生字段
        df["department"] = df["data_type"].map(TYPE_DEPT_MAP).fillna("laboratory")
        df["data_type_cn"] = df["data_type"].map(DATA_TYPE_CN).fillna("未知")

        # 保存为 Parquet (压缩率高、加载快)
        df.to_parquet(SAMPLE_PATH, index=False, compression="snappy")
        self._sample_df = df
        print(f"[DataLoader] 采样完成: {len(df):,} 条 → {SAMPLE_PATH}")
        return df

    # ---------------------------
    # 计算数据集统计信息
    # ---------------------------
    def compute_stats(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """计算数据集统计信息"""
        import json

        if self._stats is not None and not force_rebuild:
            return self._stats

        if os.path.exists(STATS_PATH) and not force_rebuild:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                self._stats = json.load(f)
            return self._stats

        df = self.load_sample(force_rebuild=force_rebuild)

        stats = {
            "total_rows": 5_000_000,  # 真实数据集总行数
            "sample_size": len(df),
            "level_distribution": df["token_level"].value_counts().to_dict(),
            "category_distribution": df["category"].value_counts().to_dict(),
            "data_type_distribution": df["data_type"].value_counts().to_dict(),
            "quality_stats": {
                "data_quality_score": {
                    "min": float(df["data_quality_score"].min()),
                    "max": float(df["data_quality_score"].max()),
                    "mean": float(df["data_quality_score"].mean()),
                    "std": float(df["data_quality_score"].std()),
                },
                "completeness": {
                    "min": float(df["completeness"].min()),
                    "max": float(df["completeness"].max()),
                    "mean": float(df["completeness"].mean()),
                },
                "accuracy": {
                    "min": float(df["accuracy"].min()),
                    "max": float(df["accuracy"].max()),
                    "mean": float(df["accuracy"].mean()),
                },
                "timeliness": {
                    "min": float(df["timeliness"].min()),
                    "max": float(df["timeliness"].max()),
                    "mean": float(df["timeliness"].mean()),
                },
                "compliance_score": {
                    "min": float(df["compliance_score"].min()),
                    "max": float(df["compliance_score"].max()),
                    "mean": float(df["compliance_score"].mean()),
                },
            },
            "by_department": {},
        }

        # 按科室统计
        for dept, group in df.groupby("department"):
            stats["by_department"][dept] = {
                "count": len(group),
                "level_a_count": int((group["token_level"] == "A").sum()),
                "level_b_count": int((group["token_level"] == "B").sum()),
                "avg_quality": float(group["data_quality_score"].mean()),
                "avg_completeness": float(group["completeness"].mean()),
                "avg_accuracy": float(group["accuracy"].mean()),
                "avg_timeliness": float(group["timeliness"].mean()),
                "avg_compliance": float(group["compliance_score"].mean()),
            }

        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        self._stats = stats
        return stats

    # ---------------------------
    # 检索相似数据 (基于真实采样)
    # ---------------------------
    def search_similar(
        self,
        quality_profile: Dict[str, float],
        department: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """基于质量画像检索相似的真实数据"""
        df = self.load_sample()

        # 过滤科室
        if department:
            df = df[df["department"] == department]

        if df.empty:
            return []

        # 计算欧氏距离
        target = np.array([
            quality_profile.get("completeness", 0),
            quality_profile.get("accuracy", 0),
            quality_profile.get("timeliness", 0),
            quality_profile.get("compliance", 0),
        ])

        candidates = df[["completeness", "accuracy", "timeliness", "compliance_score"]].values
        distances = np.linalg.norm(candidates - target, axis=1)
        similarities = 1 / (1 + distances / 100)

        # 取 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            results.append({
                "token_id": row["token_id"],
                "department": row["department"],
                "data_type": row["data_type"],
                "data_type_cn": DATA_TYPE_CN.get(row["data_type"], "未知"),
                "token_level": row["token_level"],
                "quality_score": float(row["data_quality_score"]),
                "similarity": float(round(similarities[idx], 4)),
                "dimension_scores": {
                    "completeness": float(row["completeness"]),
                    "accuracy": float(row["accuracy"]),
                    "timeliness": float(row["timeliness"]),
                    "compliance": float(row["compliance_score"]),
                },
            })
        return results

    # ---------------------------
    # 随机采样真实记录
    # ---------------------------
    def sample_records(
        self,
        n: int = 10,
        department: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """从真实数据集中随机采样记录"""
        df = self.load_sample()

        if department:
            df = df[df["department"] == department]
        if level:
            df = df[df["token_level"] == level]

        if df.empty:
            return []

        n = min(n, len(df))
        sampled = df.sample(n=n, random_state=None)
        return sampled.to_dict(orient="records")


# ============================================================
# 命令行入口 — 生成采样文件
# ============================================================
if __name__ == "__main__":
    loader = DataLoader.instance()
    df = loader.load_sample(force_rebuild=True)
    stats = loader.compute_stats(force_rebuild=True)

    print(f"\n=== 数据集统计 ===")
    print(f"总行数: {stats['total_rows']:,}")
    print(f"采样数: {stats['sample_size']:,}")
    print(f"\n等级分布:")
    for k, v in stats["level_distribution"].items():
        print(f"  {k}: {v:,}")
    print(f"\n科室分布:")
    for k, v in stats["category_distribution"].items():
        print(f"  {k}: {v:,}")
    print(f"\n质量分统计:")
    for k, v in stats["quality_stats"].items():
        print(f"  {k}: min={v['min']:.2f}, max={v['max']:.2f}, mean={v['mean']:.2f}")
