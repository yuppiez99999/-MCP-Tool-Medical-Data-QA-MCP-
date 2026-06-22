# -*- coding: utf-8 -*-
"""
models / __init__.py
模型层模块公共入口
"""

# 采用懒加载，避免导入时必须依赖 torch
__all__ = [
    "HealthcareTokenClassifier",
    "AssetValuationEngine",
    "MultimodalResearchFramework",
]


def __getattr__(name: str):
    if name == "HealthcareTokenClassifier":
        from models.classifier import HealthcareTokenClassifier
        return HealthcareTokenClassifier
    if name == "AssetValuationEngine":
        from models.valuation_engine import AssetValuationEngine
        return AssetValuationEngine
    if name == "MultimodalResearchFramework":
        from models.research_framework import MultimodalResearchFramework
        return MultimodalResearchFramework
    raise AttributeError(f"module 'models' has no attribute {name!r}")
