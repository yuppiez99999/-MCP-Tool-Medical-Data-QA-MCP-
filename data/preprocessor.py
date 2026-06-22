# -*- coding: utf-8 -*-
"""
data / preprocessor.py
医疗健康 Token 特征工程与预处理

核心功能：
- 数据清洗（缺失值、异常值、类型标准化）
- 特征工程：类别 one-hot、质量分标准化、entity_id 哈希
- 输出可直接送入 HealthcareTokenClassifier 训练
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class HealthcareTokenPreprocessor:
    """医疗 Token 特征处理器 —— 面向多任务学习的特征输出"""

    # 输出特征维度（与 config.yaml 中 model.classifier.input_dim 对齐）
    # 8 个科室 one-hot + 8 个数据类型 one-hot + 4 个质量分
    # + 4 个衍生统计 = 共 24 维（可自动扩展）
    NUMERIC_FEATURES = [
        "completeness",
        "accuracy",
        "timeliness",
        "compliance_score",
    ]

    def __init__(self, categories: List[str], data_types: List[str]):
        self.categories = list(categories)
        self.data_types = list(data_types)
        # 映射表（预测时用）
        self.cat2idx = {c: i for i, c in enumerate(self.categories)}
        self.dt2idx = {d: i for i, d in enumerate(self.data_types)}
        # 数值标准化
        self.scaler = StandardScaler()
        self._fitted = False

    # ---------------------------
    # 数据清洗
    # ---------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 缺失值填充（数值取中位数，类别取众数）
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 95.0)
        if "data_quality_score" in df.columns:
            df["data_quality_score"] = pd.to_numeric(df["data_quality_score"], errors="coerce")
            df["data_quality_score"] = df["data_quality_score"].fillna(df["data_quality_score"].median())
        if "category" in df.columns:
            df["category"] = df["category"].astype(str).str.strip().str.lower()
        if "data_type" in df.columns:
            df["data_type"] = df["data_type"].astype(str).str.strip().str.lower()
        if "token_level" in df.columns:
            df["token_level"] = df["token_level"].astype(str).str.strip().str.upper()
        return df

    # ---------------------------
    # 特征工程（训练用）
    # ---------------------------
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        df = self.clean(df)
        X = self._build_features(df)
        # 数值标准化（仅对 NUMERIC_FEATURES 对应的部分做标准化，为简化对全量做）
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        y = self._build_targets(df)
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self.clean(df)
        X = self._build_features(df)
        if self._fitted:
            return self.scaler.transform(X)
        return X

    # ---------------------------
    # 内部特征构建
    # ---------------------------
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)

        # 1) category one-hot
        cat_oh = np.zeros((n, len(self.categories)), dtype=np.float32)
        if "category" in df.columns:
            for i, val in enumerate(df["category"].tolist()):
                idx = self.cat2idx.get(val)
                if idx is not None:
                    cat_oh[i, idx] = 1.0

        # 2) data_type one-hot
        dt_oh = np.zeros((n, len(self.data_types)), dtype=np.float32)
        if "data_type" in df.columns:
            for i, val in enumerate(df["data_type"].tolist()):
                idx = self.dt2idx.get(val)
                if idx is not None:
                    dt_oh[i, idx] = 1.0

        # 3) numeric features
        num = np.zeros((n, len(self.NUMERIC_FEATURES)), dtype=np.float32)
        for j, col in enumerate(self.NUMERIC_FEATURES):
            if col in df.columns:
                num[:, j] = df[col].to_numpy(dtype=np.float32)

        # 4) 衍生统计：均值 / 方差 / 最高 / 最低
        num_mean = num.mean(axis=1, keepdims=True)
        num_std = num.std(axis=1, keepdims=True)
        num_max = num.max(axis=1, keepdims=True)
        num_min = num.min(axis=1, keepdims=True)

        X = np.concatenate(
            [cat_oh, dt_oh, num, num_mean, num_std, num_max, num_min],
            axis=1,
        )
        return X.astype(np.float32)

    def _build_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        y = {}
        # 等级二分类：A = 1，B = 0
        if "token_level" in df.columns:
            y["level"] = (df["token_level"].values == "A").astype(np.int64)
        # 质量分回归
        if "data_quality_score" in df.columns:
            y["quality"] = df["data_quality_score"].to_numpy(dtype=np.float32)
        # 科室多分类
        if "category" in df.columns:
            cat_idx = np.array(
                [self.cat2idx.get(v, 0) for v in df["category"].tolist()],
                dtype=np.int64,
            )
            y["category"] = cat_idx
        return y

    # ---------------------------
    # 单条 token -> 特征向量（预测时用）
    # ---------------------------
    def token_to_features(self, row: Dict) -> np.ndarray:
        df = pd.DataFrame([row])
        return self.transform(df)[0]

    @property
    def input_dim(self) -> int:
        """输出特征维度（category + data_type + numeric + 4 stats）"""
        return len(self.categories) + len(self.data_types) + len(self.NUMERIC_FEATURES) + 4
