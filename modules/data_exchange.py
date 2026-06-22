# -*- coding: utf-8 -*-
"""
modules / data_exchange.py
北数所数据产品登记集成模块

功能：
1. 根据数据集自动生成符合数据资产登记规范的完整报告
2. 包含产品元信息、合规声明、质量指标、价值估算、字段字典、样本展示
3. 输出为 Markdown / JSON，便于向交易所提交或内部归档
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


class DataExchangeRegistrar:
    """北数所数据产品登记集成模块"""

    def __init__(self, config: Dict, dataset_summary: Optional[Dict] = None, valuation_summary: Optional[Dict] = None):
        self.cfg = config.get("data_exchange", {})
        self.categories = config.get("categories", {})
        self.data_types = config.get("data_types", {})
        self.dataset_summary = dataset_summary or {}
        self.valuation_summary = valuation_summary or {}
        self.registration_prefix = self.cfg.get("registration_prefix", "BJDE-HEALTH")

    # ---------------------------
    # 产品元信息
    # ---------------------------
    def product_meta(self) -> Dict:
        return {
            "exchange_name": self.cfg.get("exchange_name", "北京国际大数据交易所"),
            "product_name": self.cfg.get("product_name", "医疗健康 A 级 / B 级 Token 数据集"),
            "data_provider": self.cfg.get("data_provider", "医疗数据联盟"),
            "data_format": self.cfg.get("data_format", "CSV / JSON"),
            "registration_id": f"{self.registration_prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_scale": f"{int(self.dataset_summary.get('total_rows', 0)):,} 条 Token",
        }

    # ---------------------------
    # 质量指标
    # ---------------------------
    def quality_metrics(self) -> Dict:
        return {
            "quality_min": float(self.dataset_summary.get("quality_min", 0)),
            "quality_max": float(self.dataset_summary.get("quality_max", 0)),
            "quality_mean": float(self.dataset_summary.get("quality_mean", 0)),
            "level_distribution": self.dataset_summary.get("level_counts", {}),
            "category_distribution": self.dataset_summary.get("category_counts", {}),
        }

    # ---------------------------
    # 价值估算
    # ---------------------------
    def value_metrics(self) -> Dict:
        return {
            "total_tokens": int(self.valuation_summary.get("total_tokens", 0)),
            "total_value": float(self.valuation_summary.get("total_value", 0)),
            "total_enterprise_value": float(
                self.valuation_summary.get("total_enterprise_value", 0)
            ),
            "avg_value_per_token": float(self.valuation_summary.get("avg_value_per_token", 0)),
            "by_category": self.valuation_summary.get("by_category", []),
            "by_level": self.valuation_summary.get("by_level", []),
        }

    # ---------------------------
    # 字段字典
    # ---------------------------
    def data_dictionary(self) -> Dict[str, Dict]:
        return {
            "token_id": {
                "type": "string",
                "description": "Token 唯一标识，用于合规审计与溯源",
                "sample": "HEALTH-XXXX-XXXX",
            },
            "domain": {
                "type": "string",
                "description": "数据所属领域",
                "sample": "healthcare",
            },
            "category": {
                "type": "string",
                "description": "医疗科室类别（放射科/检验科/病理科/心血管科/神经内科/骨科/儿科/急诊科）",
                "sample": "radiology",
            },
            "data_type": {
                "type": "string",
                "description": "数据形态（影像/文本/心电/检验/病理/基因/生命体征等）",
                "sample": "image",
            },
            "entity_id": {
                "type": "string",
                "description": "脱敏后的实体标识，支持同一实体跨科室聚合",
                "sample": "ENT-XXXX",
            },
            "data_quality_score": {
                "type": "float",
                "description": "综合数据质量分（0-100）",
                "sample": 98.5,
            },
            "token_level": {
                "type": "string",
                "description": "Token 等级：A 高质量 / B 标准质量",
                "sample": "A",
            },
            "completeness": {
                "type": "float",
                "description": "数据完整性分（0-100）",
                "sample": 99.0,
            },
            "accuracy": {
                "type": "float",
                "description": "数据准确性分（0-100）",
                "sample": 97.0,
            },
            "timeliness": {
                "type": "float",
                "description": "时效性分（0-100）",
                "sample": 95.0,
            },
            "compliance_score": {
                "type": "float",
                "description": "合规性分（0-100），≥95 视为资产合格",
                "sample": 100.0,
            },
            "created_at": {
                "type": "datetime",
                "description": "Token 创建时间",
                "sample": "2026-06-16 08:00:00",
            },
        }

    # ---------------------------
    # 样本展示
    # ---------------------------
    def sample_records(self, df: pd.DataFrame, n: int = 5) -> Dict:
        cols = [c for c in [
            "token_id", "category", "data_type", "token_level",
            "data_quality_score", "completeness", "accuracy",
            "timeliness", "compliance_score", "created_at",
        ] if c in df.columns]
        sample = df[cols].head(n).to_dict(orient="records") if not df.empty else []
        return {"count": len(sample), "records": sample}

    # ---------------------------
    # 生成 Markdown 报告
    # ---------------------------
    def generate_markdown(
        self,
        raw_df_sample: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> str:
        meta = self.product_meta()
        quality = self.quality_metrics()
        value = self.value_metrics()
        dic = self.data_dictionary()
        sample_df = raw_df_sample if raw_df_sample is not None else pd.DataFrame()
        sample = self.sample_records(sample_df, n=5)
        compliance_statement = self.cfg.get(
            "compliance_statement",
            "本数据集严格遵循数据安全法等相关法律法规。",
        )

        lines: list[str] = []
        lines.append(f"# 数据资产登记报告 · {meta['product_name']}")
        lines.append("")
        lines.append(f"- **登记 ID**: {meta['registration_id']}")
        lines.append(f"- **登记机构**: {meta['exchange_name']}")
        lines.append(f"- **数据提供方**: {meta['data_provider']}")
        lines.append(f"- **数据格式**: {meta['data_format']}")
        lines.append(f"- **数据集规模**: {meta['dataset_scale']}")
        lines.append(f"- **登记时间**: {meta['registered_at']}")
        lines.append("")

        lines.append("## 一、合规性声明")
        lines.append("")
        lines.append(compliance_statement.strip())
        lines.append("")

        lines.append("## 二、数据质量指标")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 质量分范围 | {quality['quality_min']:.2f} - {quality['quality_max']:.2f} |")
        lines.append(f"| 质量分均值 | {quality['quality_mean']:.2f} |")
        lvl = quality["level_distribution"]
        total = max(1, sum(lvl.values()))
        for k, v in lvl.items():
            lines.append(f"| 等级 {k} 占比 | {int(v)} ({100.0 * v / total:.2f}%) |")
        lines.append("")

        # 科室分布表
        lines.append("### 科室 Token 分布")
        lines.append("")
        lines.append("| 科室 (category) | 中文名 | Token 数 | 占比 |")
        lines.append("|------------------|-------|---------|------|")
        cat_dist = quality.get("category_distribution", {})
        total_cat = max(1, sum(cat_dist.values()))
        for key, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
            name_cn = self.categories.get(key, {}).get("name_cn", key)
            lines.append(f"| {key} | {name_cn} | {int(count)} | {100.0 * count / total_cat:.2f}% |")
        lines.append("")

        lines.append("## 三、数据资产价值估算")
        lines.append("")
        lines.append(f"- 合格 Token 数：{value['total_tokens']:,}")
        lines.append(f"- 总价值（单条参考价）：¥ {value['total_value']:,.2f}")
        lines.append(f"- 总价值（企业批量折扣价）：¥ {value['total_enterprise_value']:,.2f}")
        lines.append(f"- 平均每条价值：¥ {value['avg_value_per_token']:.2f}")
        lines.append("")

        # 按科室价值
        if value.get("by_category"):
            lines.append("### 按科室价值分布")
            lines.append("")
            lines.append("| 科室 | category | count | total_value | avg_value |")
            lines.append("|------|----------|-------|-------------|-----------|")
            for row in value["by_category"]:
                lines.append(
                    f"| {row.get('category_cn', '')} | {row.get('category', '')} | "
                    f"{int(row.get('count', 0))} | ¥ {float(row.get('total_value', 0)):,.2f} | ¥ {float(row.get('avg_value', 0)):,.2f} |"
                )
            lines.append("")

        lines.append("## 四、数据字段字典")
        lines.append("")
        lines.append("| 字段 | 类型 | 说明 | 示例 |")
        lines.append("|------|------|------|------|")
        for key, meta_field in dic.items():
            lines.append(
                f"| {key} | {meta_field['type']} | {meta_field['description']} | `{meta_field['sample']}` |"
            )
        lines.append("")

        if sample["records"]:
            lines.append("## 五、数据样本展示")
            lines.append("")
            keys = list(sample["records"][0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
            for r in sample["records"]:
                lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
            lines.append("")

        lines.append("---")
        lines.append(f"*本报告由 DataExchangeRegistrar 于 {meta['registered_at']} 自动生成*")
        md_text = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_text)
        return md_text

    # ---------------------------
    # 生成 JSON 摘要（用于 API 接入）
    # ---------------------------
    def generate_json(
        self,
        raw_df_sample: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> str:
        payload = {
            "meta": self.product_meta(),
            "quality": self.quality_metrics(),
            "value": self.value_metrics(),
            "dictionary": self.data_dictionary(),
            "sample": self.sample_records(raw_df_sample if raw_df_sample is not None else pd.DataFrame(), n=5),
            "compliance_statement": self.cfg.get("compliance_statement", ""),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def __repr__(self) -> str:
        return f"DataExchangeRegistrar(prefix={self.registration_prefix})"
