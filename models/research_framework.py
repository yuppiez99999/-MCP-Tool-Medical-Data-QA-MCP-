# -*- coding: utf-8 -*-
"""
models / research_framework.py
多模态医学研究框架

核心能力：
1. 跨科室 Token 聚合分析 —— 同一 entity_id 在不同科室的 Token 分布
2. 质量分布对比研究 —— 不同科室 / 模态之间的质量差异统计
3. 数据价值时间序列分析 —— 创建时间对估值的影响
4. Token 图谱构建 —— 基于实体 × 科室的关联网络（邻接矩阵）
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class MultimodalResearchFramework:
    """多模态医学研究框架 —— 围绕 8 大科室的跨学科研究能力"""

    def __init__(self, df: Optional[pd.DataFrame] = None, categories: Optional[List[str]] = None):
        self.df = df if df is not None else pd.DataFrame()
        self.categories = categories or []

    # ---------------------------
    # 1) 跨科室实体聚合
    # ---------------------------
    def entity_multi_department(self, top_n: int = 50) -> pd.DataFrame:
        """对每个 entity_id，统计其在不同科室出现的 Token 数量、质量分均值、估值"""
        if self.df.empty or "entity_id" not in self.df.columns:
            return pd.DataFrame()

        df = self.df.copy()
        # 确保有 category / data_quality_score
        required = {"entity_id", "category", "data_quality_score"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"缺少必要列: {missing}")

        agg = (
            df.groupby(["entity_id", "category"])
            .agg(
                token_count=("token_id", "nunique"),
                avg_quality=("data_quality_score", "mean"),
                min_quality=("data_quality_score", "min"),
                max_quality=("data_quality_score", "max"),
            )
            .reset_index()
        )
        # 按 Token 数排序
        top_entities = (
            df.groupby("entity_id")["token_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        return agg[agg["entity_id"].isin(top_entities)].sort_values(
            ["entity_id", "token_count"], ascending=[True, False]
        )

    # ---------------------------
    # 2) 质量分布对比
    # ---------------------------
    def quality_by_category(self) -> pd.DataFrame:
        """按科室 / 数据类型的质量分布统计"""
        if self.df.empty:
            return pd.DataFrame()
        cols = set(self.df.columns)
        group_cols = [c for c in ["category", "data_type", "token_level"] if c in cols]
        if not group_cols:
            return pd.DataFrame()
        if "data_quality_score" not in cols:
            return pd.DataFrame()

        return (
            self.df.groupby(group_cols)["data_quality_score"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "token_count",
                    "mean": "avg_quality",
                    "std": "quality_std",
                }
            )
        )

    # ---------------------------
    # 3) 时间序列分析
    # ---------------------------
    def time_series_value(
        self,
        value_series: Optional[pd.Series] = None,
        freq: str = "W",
    ) -> pd.DataFrame:
        """
        按日期聚合 Token 数量和价值趋势
        :param value_series: 可选，与 self.df 对齐的价值序列（例如单 Token 价值）
        :param freq: 'D' 日, 'W' 周, 'M' 月
        """
        if self.df.empty or "created_at" not in self.df.columns:
            return pd.DataFrame()
        df = self.df.copy()
        try:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        except Exception:
            return pd.DataFrame()
        if value_series is not None:
            df["value"] = value_series.values
        else:
            # 无外部价值序列时，以质量分作为价值代理
            if "data_quality_score" in df.columns:
                df["value"] = df["data_quality_score"].astype(float)
            else:
                df["value"] = 1.0

        df = df.dropna(subset=["created_at"])
        df = df.set_index("created_at")
        agg = df.resample(freq).agg(token_count=("value", "size"), total_value=("value", "sum"), avg_value=("value", "mean"))
        agg = agg.reset_index()
        return agg

    # ---------------------------
    # 4) Token 图谱：实体 × 科室邻接矩阵
    # ---------------------------
    def entity_category_adjacency(self, top_entities: int = 100) -> Dict:
        """构建实体-科室邻接矩阵，可用于进一步图分析（PageRank / 社区发现）"""
        if self.df.empty or not {"entity_id", "category"}.issubset(set(self.df.columns)):
            return {}

        # 取 top_entities
        top = (
            self.df["entity_id"]
            .value_counts()
            .head(top_entities)
            .index.tolist()
        )
        sub = self.df[self.df["entity_id"].isin(top)]

        # pivot: 行 = entity_id, 列 = category, 值 = token 数
        matrix = sub.groupby(["entity_id", "category"]).size().unstack(fill_value=0)

        # 相关矩阵（实体间通过科室的共现）
        corr = matrix.T.corr()
        # 科室分布相似度
        category_distribution = matrix.sum(axis=0).sort_values(ascending=False)

        return {
            "entities": top,
            "categories": list(matrix.columns),
            "adjacency_matrix": matrix,
            "entity_correlation": corr,
            "category_distribution": category_distribution,
        }

    # ---------------------------
    # 综合研究报告
    # ---------------------------
    def research_report(self, category: Optional[str] = None) -> Dict:
        """生成某个科室 / 全量数据集的综合研究报告"""
        df = self.df
        if category and "category" in df.columns:
            df = df[df["category"] == category]
        report: Dict = {
            "category": category or "ALL",
            "token_count": int(len(df)),
        }
        if "data_quality_score" in df.columns:
            q = df["data_quality_score"].astype(float).describe()
            report["quality_stats"] = q.to_dict()
        if "token_level" in df.columns:
            report["level_distribution"] = df["token_level"].value_counts().to_dict()
        if "data_type" in df.columns:
            report["data_type_distribution"] = df["data_type"].value_counts().to_dict()
        if "entity_id" in df.columns:
            report["unique_entities"] = int(df["entity_id"].nunique())
        return report

    def __repr__(self) -> str:
        return f"MultimodalResearchFramework(rows={len(self.df)}, categories={self.categories})"
