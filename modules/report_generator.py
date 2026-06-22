# -*- coding: utf-8 -*-
"""
modules / report_generator.py
报告生成系统

核心功能：
1. 月度数据质量报告
2. 季度研究分析报告
3. API 使用统计报告
4. 审计合规报告

报告格式：Markdown / HTML / PDF
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from data.loader import HealthcareTokenLoader
from models.valuation_engine import AssetValuationEngine
from modules.audit_trail import AuditTrailModule
from modules.auth_manager import AuthManager


class ReportGenerator:
    """报告生成系统"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.yaml",
            )
        self.config_path = config_path
        self.loader = HealthcareTokenLoader(config_path=config_path)
        self.valuation_engine = AssetValuationEngine(config_path=config_path)
        self.audit = AuditTrailModule()
        self.auth = AuthManager(config_path=config_path)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "outputs", "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_category_name(self, category: str) -> str:
        categories = self.loader.categories
        return categories.get(category, {}).get("name_cn", category)

    def generate_quality_report(self, period: str = "monthly") -> str:
        """生成数据质量报告"""
        now = datetime.now()
        if period == "weekly":
            start_date = now - timedelta(days=7)
        elif period == "monthly":
            start_date = now.replace(day=1)
        elif period == "quarterly":
            quarter = (now.month - 1) // 3
            start_date = now.replace(month=quarter * 3 + 1, day=1)
        else:
            start_date = now - timedelta(days=30)

        summary = self.loader.summary()
        df_sample = self.loader.load_all(sample=5000)

        if not df_sample.empty:
            valued = self.valuation_engine.value_dataframe(df_sample)
            val_summary = self.valuation_engine.summary(valued)
        else:
            val_summary = {}

        lines = []
        lines.append(f"# 数据质量报告 · {period}")
        lines.append("")
        lines.append(f"- **生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **统计周期**: {start_date.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}")
        lines.append(f"- **数据集规模**: {summary.get('total_rows', 0):,} 条")
        lines.append("")

        lines.append("## 一、质量概况")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 质量分均值 | {summary.get('quality_mean', 0):.2f} |")
        lines.append(f"| 质量分最小值 | {summary.get('quality_min', 0):.2f} |")
        lines.append(f"| 质量分最大值 | {summary.get('quality_max', 0):.2f} |")
        lines.append("")

        level_counts = summary.get("level_counts", {})
        total_level = sum(level_counts.values()) or 1
        lines.append("### 等级分布")
        lines.append("")
        lines.append("| 等级 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for level in ["A", "B"]:
            count = level_counts.get(level, 0)
            lines.append(f"| {level} | {count:,} | {count / total_level * 100:.2f}% |")
        lines.append("")

        category_counts = summary.get("category_counts", {})
        total_category = sum(category_counts.values()) or 1
        lines.append("### 科室分布")
        lines.append("")
        lines.append("| 科室 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            name_cn = self._get_category_name(cat)
            lines.append(f"| {name_cn} | {count:,} | {count / total_category * 100:.2f}% |")
        lines.append("")

        if val_summary:
            lines.append("## 二、资产估值摘要")
            lines.append("")
            lines.append(f"- 合格 Token 数：{val_summary.get('total_tokens', 0):,}")
            lines.append(f"- 总价值：¥ {val_summary.get('total_value', 0):,.2f}")
            lines.append(f"- 平均每条价值：¥ {val_summary.get('avg_value_per_token', 0):.2f}")
            lines.append("")

            by_category = val_summary.get("by_category", [])
            if by_category:
                lines.append("### 按科室价值分布")
                lines.append("")
                lines.append("| 科室 | 数量 | 总价值 | 平均价值 |")
                lines.append("|------|------|---------|----------|")
                for row in by_category:
                    lines.append(
                        f"| {row.get('category_cn', '')} | {int(row.get('count', 0)):,} | "
                        f"¥ {float(row.get('total_value', 0)):,.2f} | "
                        f"¥ {float(row.get('avg_value', 0)):.2f} |"
                    )
                lines.append("")

        audit_stats = self.audit.statistics(days=30)
        lines.append("## 三、审计统计")
        lines.append("")
        lines.append(f"- 近 30 天审计日志数：{audit_stats.get('total_logs_last_n_days', 0):,}")
        lines.append(f"- 高风险日志数：{audit_stats.get('high_risk_logs', 0)}")
        lines.append("")

        by_access_type = audit_stats.get("by_access_type", {})
        if by_access_type:
            lines.append("### 访问类型分布")
            lines.append("")
            lines.append("| 类型 | 数量 |")
            lines.append("|------|------|")
            for access_type, count in by_access_type.items():
                lines.append(f"| {access_type} | {count:,} |")
            lines.append("")

        report_content = "\n".join(lines)

        filename = f"quality_report_{period}_{now.strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content

    def generate_research_report(self, category: Optional[str] = None, sample: int = 10000) -> str:
        """生成研究分析报告"""
        from models.research_framework import MultimodalResearchFramework

        now = datetime.now()
        df = self.loader.filter_by(category=category, limit=sample)

        if df.empty:
            return f"# 研究分析报告\n\n无数据可用"

        framework = MultimodalResearchFramework(df=df, categories=list(self.loader.categories.keys()))
        research_report = framework.research_report(category=category)

        lines = []
        lines.append(f"# 研究分析报告")
        lines.append("")
        lines.append(f"- **生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **分析科室**: {self._get_category_name(category) if category else '全部'}")
        lines.append(f"- **分析样本量**: {len(df):,} 条")
        lines.append("")

        lines.append("## 一、研究摘要")
        lines.append("")
        for key, value in research_report.items():
            if isinstance(value, (int, float)):
                lines.append(f"- **{key}**: {value}")
            elif isinstance(value, dict):
                lines.append(f"- **{key}**:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")

        lines.append("## 二、质量分布")
        lines.append("")
        quality_df = framework.quality_by_category()
        if not quality_df.empty:
            lines.append("| 科室 | 样本数 | 质量均值 |")
            lines.append("|------|--------|----------|")
            for _, row in quality_df.head(10).iterrows():
                lines.append(f"| {row.get('category_cn', '')} | {int(row.get('count', 0))} | {row.get('quality_mean', 0):.2f} |")
            lines.append("")

        lines.append("## 三、数据统计")
        lines.append("")
        numeric_cols = ["data_quality_score", "completeness", "accuracy", "timeliness", "compliance_score"]
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    lines.append(f"- **{col}**: 均值 {values.mean():.2f}, 最小 {values.min():.2f}, 最大 {values.max():.2f}")
        lines.append("")

        report_content = "\n".join(lines)

        filename = f"research_report_{category or 'all'}_{now.strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content

    def generate_api_stats_report(self, api_key: str, days: int = 30) -> str:
        """生成 API 使用统计报告"""
        now = datetime.now()
        stats = self.auth.get_user_stats(api_key, days)
        user_info = self.auth.get_user_info(api_key)

        if not stats or not user_info:
            return "# API 使用统计报告\n\n无效的 API 密钥"

        lines = []
        lines.append("# API 使用统计报告")
        lines.append("")
        lines.append(f"- **生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **统计周期**: {days} 天")
        lines.append(f"- **用户**: {user_info.get('username', '')}")
        lines.append(f"- **套餐**: {user_info.get('plan_name', '')}")
        lines.append("")

        lines.append("## 一、使用概况")
        lines.append("")
        lines.append(f"- 总调用次数：{stats.get('total_calls', 0):,}")
        lines.append(f"- 总费用：¥ {stats.get('total_cost', 0):.2f}")
        lines.append(f"- 日均调用：{stats.get('total_calls', 0) / max(1, days):.1f}")
        lines.append("")

        lines.append("## 二、按接口统计")
        lines.append("")
        by_endpoint = stats.get("by_endpoint", {})
        if by_endpoint:
            lines.append("| 接口 | 调用次数 | 费用 |")
            lines.append("|------|----------|------|")
            for endpoint, data in by_endpoint.items():
                lines.append(f"| {endpoint} | {data.get('calls', 0):,} | ¥ {data.get('cost', 0):.2f} |")
            lines.append("")

        lines.append("## 三、套餐使用情况")
        lines.append("")
        lines.append(f"- 套餐日限制：{user_info.get('daily_limit', 0)} 次")
        lines.append(f"- 当前日调用：{user_info.get('daily_calls', 0)} 次")
        lines.append(f"- 剩余日调用：{max(0, user_info.get('daily_limit', 0) - user_info.get('daily_calls', 0))} 次")
        lines.append(f"- 订阅到期：{datetime.fromtimestamp(user_info.get('subscription_end', 0)).strftime('%Y-%m-%d')}")
        lines.append("")

        report_content = "\n".join(lines)

        filename = f"api_stats_report_{user_info.get('user_id', '')}_{now.strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content

    def generate_audit_report(self, days: int = 30) -> str:
        """生成审计合规报告"""
        now = datetime.now()
        stats = self.audit.statistics(days)
        anomalies = self.audit.detect_anomalies(minutes=60)
        high_risk = self.audit.high_risk_logs(limit=50)

        lines = []
        lines.append("# 审计合规报告")
        lines.append("")
        lines.append(f"- **生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **统计周期**: {days} 天")
        lines.append("")

        lines.append("## 一、审计概况")
        lines.append("")
        lines.append(f"- 审计日志总数：{stats.get('total_logs_last_n_days', 0):,}")
        lines.append(f"- 高风险日志数：{stats.get('high_risk_logs', 0)}")
        lines.append(f"- 异常阈值：{stats.get('anomaly_threshold', 0)}")
        lines.append("")

        by_access_type = stats.get("by_access_type", {})
        if by_access_type:
            lines.append("### 访问类型分布")
            lines.append("")
            total = sum(by_access_type.values()) or 1
            lines.append("| 类型 | 数量 | 占比 |")
            lines.append("|------|------|------|")
            for access_type, count in sorted(by_access_type.items(), key=lambda x: -x[1]):
                lines.append(f"| {access_type} | {count:,} | {count / total * 100:.2f}% |")
            lines.append("")

        lines.append("## 二、异常检测")
        lines.append("")
        if anomalies:
            lines.append("| 访问者 | 调用次数 | 最高风险评分 |")
            lines.append("|--------|----------|--------------|")
            for anomaly in anomalies:
                lines.append(f"| {anomaly.get('accessor_id', '')} | {anomaly.get('cnt', 0):,} | {anomaly.get('max_risk', 0):.2f} |")
            lines.append("")
        else:
            lines.append("无异常检测结果")
            lines.append("")

        lines.append("## 三、高风险日志")
        lines.append("")
        if high_risk:
            lines.append("| 时间 | Token ID | 访问者 | 类型 | 风险评分 |")
            lines.append("|------|----------|--------|------|----------|")
            for log in high_risk[:10]:
                lines.append(
                    f"| {log.get('access_datetime', '')} | {log.get('token_id', '')} | "
                    f"{log.get('accessor_id', '')} | {log.get('access_type', '')} | "
                    f"{log.get('risk_score', 0):.2f} |"
                )
            lines.append("")
        else:
            lines.append("无高风险日志")
            lines.append("")

        report_content = "\n".join(lines)

        filename = f"audit_report_{now.strftime('%Y%m%d')}.md"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content

    def list_reports(self) -> List[Dict]:
        """列出所有已生成的报告"""
        reports = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.output_dir, filename)
                mtime = os.path.getmtime(filepath)
                reports.append({
                    "filename": filename,
                    "path": filepath,
                    "size": os.path.getsize(filepath),
                    "modified_at": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
        return sorted(reports, key=lambda x: x["modified_at"], reverse=True)

    def get_report(self, filename: str) -> Optional[str]:
        """获取指定报告内容"""
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def __repr__(self) -> str:
        return f"ReportGenerator(output_dir={self.output_dir})"
