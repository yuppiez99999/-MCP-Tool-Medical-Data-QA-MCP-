# -*- coding: utf-8 -*-
"""
models / valuation_engine.py
数据资产估值引擎 —— 基于 Token 的多维价值评估

公式：
  Token 价值 = 基础价值 × 质量系数 × 类别权重 × 稀有度系数 × 等级溢价
  其中：
    - 基础价值 = 所在科室的基准单价（config.yaml 中 categories[...].base_price）
    - 质量系数 = (data_quality_score / 100)^quality_exponent
    - 类别权重 = 科室配置中的 weight × 数据类型配置中的 weight
    - 稀有度系数 = 1 / (该类别 Token 占比 ^ rarity_exponent)
    - 等级溢价 = A 级 1.20，B 级 1.00

支持批量估值 / 科室汇总 / 价值区间估算。
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


# -----------------------------
# 配置加载
# -----------------------------
def _load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------
# 估值引擎
# -----------------------------
class AssetValuationEngine:
    """数据资产估值引擎 —— 面向医疗机构数据入表定价基准"""

    def __init__(self, config_path: Optional[str] = None, category_counts: Optional[Dict[str, int]] = None):
        self.config = _load_config(config_path)
        self.categories = self.config["categories"]
        self.data_types = self.config["data_types"]
        self.valuation_cfg = self.config["valuation"]
        # 类别占比（用于计算稀有度系数）
        self.category_counts: Dict[str, int] = category_counts or {}
        self._total_tokens = sum(self.category_counts.values()) if self.category_counts else 0

    # ---------------------------
    # 单 Token 估值
    # ---------------------------
    def value_token(self, row: Dict) -> Dict:
        category = str(row.get("category", "")).lower()
        data_type = str(row.get("data_type", "")).lower()
        level = str(row.get("token_level", "B")).upper()
        quality = float(row.get("data_quality_score", 95.0))
        compliance = float(row.get("compliance_score", 95.0))

        cat_cfg = self.categories.get(category, {"base_price": 8.0, "weight": 1.0, "name_cn": category})
        dt_cfg = self.data_types.get(data_type, {"weight": 1.0, "name_cn": data_type})

        base_price = float(cat_cfg.get("base_price", 8.0))
        cat_weight = float(cat_cfg.get("weight", 1.0))
        dt_weight = float(dt_cfg.get("weight", 1.0))

        q_exp = float(self.valuation_cfg["quality_exponent"])
        quality_coef = (quality / 100.0) ** q_exp

        # 稀有度
        r_exp = float(self.valuation_cfg["rarity_exponent"])
        if self._total_tokens > 0 and category in self.category_counts:
            ratio = max(1e-6, self.category_counts[category] / self._total_tokens)
            rarity = 1.0 / (ratio ** r_exp)
        else:
            rarity = 1.0

        level_bonus = (
            float(self.valuation_cfg["level_a_bonus"])
            if level == "A"
            else float(self.valuation_cfg["level_b_bonus"])
        )

        value = base_price * quality_coef * cat_weight * dt_weight * rarity * level_bonus

        # 合规校验：低于最低合规分的 Token 不计入资产
        min_compliance = float(self.valuation_cfg["min_compliance_score"])
        eligible = compliance >= min_compliance

        # 企业批量折扣价（用于数据入表参考）
        enterprise_price = value * float(self.valuation_cfg["enterprise_discount"])

        return {
            "token_id": row.get("token_id", ""),
            "category": category,
            "category_cn": cat_cfg.get("name_cn", category),
            "data_type": data_type,
            "token_level": level,
            "data_quality_score": quality,
            "compliance_score": compliance,
            "base_price": base_price,
            "quality_coef": round(quality_coef, 4),
            "rarity_coef": round(rarity, 4),
            "level_bonus": level_bonus,
            "single_value": round(value, 2),
            "enterprise_value": round(enterprise_price, 2),
            "eligible": eligible,
        }

    # ---------------------------
    # 批量估值（支持 3.9M 条大数据分块）
    # ---------------------------
    def value_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict] = []
        for _, row in df.iterrows():
            records.append(self.value_token(row.to_dict()))
        result = pd.DataFrame(records)
        return result

    def value_iter_chunks(self, df_iter) -> pd.DataFrame:
        """对 DataFrame 迭代器（pandas chunk）进行估值"""
        result_parts: List[pd.DataFrame] = []
        for chunk in df_iter:
            result_parts.append(self.value_dataframe(chunk))
        if not result_parts:
            return pd.DataFrame()
        return pd.concat(result_parts, ignore_index=True)

    # ---------------------------
    # 汇总统计
    # ---------------------------
    def summary(self, valued_df: pd.DataFrame) -> Dict:
        """返回科室 / 等级维度的资产价值汇总"""
        if valued_df is None or valued_df.empty:
            return {}

        # 仅统计合规 eligible 的
        eligible = valued_df[valued_df["eligible"] == True]  # noqa: E712
        # 按科室汇总
        by_category = (
            eligible.groupby(["category_cn", "category"])["single_value"]
            .agg(["count", "sum", "mean", "min", "max"])
            .reset_index()
            .rename(columns={"sum": "total_value", "mean": "avg_value"})
            .to_dict(orient="records")
        )
        # 按等级汇总
        by_level = (
            eligible.groupby("token_level")["single_value"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(columns={"sum": "total_value", "mean": "avg_value"})
            .to_dict(orient="records")
        )
        total_value = float(eligible["single_value"].sum())
        total_enterprise_value = float(eligible["enterprise_value"].sum())
        return {
            "total_tokens": int(len(eligible)),
            "total_value": round(total_value, 2),
            "total_enterprise_value": round(total_enterprise_value, 2),
            "avg_value_per_token": round(total_value / max(1, len(eligible)), 2),
            "by_category": by_category,
            "by_level": by_level,
        }

    # ---------------------------
    # 动态更新类别统计
    # ---------------------------
    def update_category_counts(self, counts: Dict[str, int]):
        self.category_counts = counts
        self._total_tokens = sum(counts.values()) if counts else 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def __repr__(self) -> str:
        return f"AssetValuationEngine(categories={len(self.categories)}, total_tokens={self._total_tokens})"
