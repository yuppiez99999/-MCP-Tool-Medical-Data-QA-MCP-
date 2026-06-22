# -*- coding: utf-8 -*-
"""
医疗数据质量评估 MCP Server (FastMCP) — 真实数据集版
=====================================================
小X宝医疗黑客松参赛作品 — ModelScope MCP 广场提交版

数据集: 5,000,000 条 A级/B级 医疗 Token (北数所合规数据)
采样: 50,000 条分层采样 (Parquet缓存，毫秒级响应)

依赖: pip install fastmcp pandas numpy pyarrow
运行: python server.py
协议: MCP (Model Context Protocol) over stdio
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 sys.path (允许导入 data.loader)
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastmcp import FastMCP
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

# 导入真实数据加载器
try:
    from data.loader import DataLoader, TYPE_DEPT_MAP, DATA_TYPE_CN
    _LOADER = DataLoader.instance()
    _HAS_REAL_DATA = True
except Exception as e:
    print(f"[WARN] 真实数据加载失败，回退到模拟模式: {e}", file=sys.stderr)
    _LOADER = None
    _HAS_REAL_DATA = False

# 创建 MCP Server
mcp = FastMCP("MedicalDataQA")

# ============================================================
# 常量定义
# ============================================================
DEPARTMENTS = {
    "radiology":    {"name_cn": "放射科", "weight": 1.50, "desc": "影像数据稀缺，AI训练需求高"},
    "pathology":    {"name_cn": "病理科", "weight": 1.40, "desc": "病理标注数据专业度高"},
    "neurology":    {"name_cn": "神经内科", "weight": 1.35, "desc": "脑电/神经影像价值"},
    "cardiology":   {"name_cn": "心血管科", "weight": 1.30, "desc": "ECG数据临床价值高"},
    "laboratory":   {"name_cn": "检验科", "weight": 1.00, "desc": "检验报告结构化"},
    "orthopedics":  {"name_cn": "骨科", "weight": 0.95, "desc": "X-Ray影像"},
    "pediatrics":   {"name_cn": "儿科", "weight": 0.90, "desc": "儿童发育数据"},
    "emergency":    {"name_cn": "急诊科", "weight": 0.85, "desc": "分诊记录"},
}

DATA_TYPES = {
    "ct_image":        {"name_cn": "CT影像", "weight": 1.40},
    "blood_test":      {"name_cn": "血液检验", "weight": 0.95},
    "pathology_slide": {"name_cn": "病理切片", "weight": 1.35},
    "ecg":             {"name_cn": "心电图", "weight": 1.30},
    "ultrasound":      {"name_cn": "超声", "weight": 1.20},
    "x_ray":           {"name_cn": "X光", "weight": 1.10},
    "growth_record":   {"name_cn": "生长记录", "weight": 0.90},
    "triage":          {"name_cn": "分诊记录", "weight": 0.85},
    # 兼容旧映射
    "image":    {"name_cn": "影像", "weight": 1.40},
    "text":     {"name_cn": "文本", "weight": 1.00},
    "lab":      {"name_cn": "检验", "weight": 0.95},
    "pathology":{"name_cn": "病理", "weight": 1.35},
    "genetic":  {"name_cn": "基因", "weight": 1.50},
    "vital":    {"name_cn": "生命体征", "weight": 0.80},
    "other":    {"name_cn": "其他", "weight": 0.70},
}

QUALITY_DIMENSIONS = ["completeness", "accuracy", "timeliness", "compliance"]


# ============================================================
# 内部辅助函数
# ============================================================
def _infer_department(record: Dict) -> str:
    """基于数据类型和内容推断科室"""
    data_type = record.get("data_type", "text")
    category = record.get("category", record.get("department", ""))
    if category and category in DEPARTMENTS:
        return category
    return TYPE_DEPT_MAP.get(data_type, "laboratory")


def _grade_level(score: float) -> str:
    """根据质量分评定等级"""
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"


def _match_score(data_type: str, dept: str) -> float:
    """数据类型与科室的匹配度"""
    match_map = {
        ("ct_image", "radiology"): 0.98, ("ct_image", "orthopedics"): 0.70,
        ("blood_test", "laboratory"): 0.95,
        ("pathology_slide", "pathology"): 0.98,
        ("ecg", "cardiology"): 0.98,
        ("ultrasound", "radiology"): 0.85,
        ("x_ray", "orthopedics"): 0.95, ("x_ray", "radiology"): 0.80,
        ("growth_record", "pediatrics"): 0.95,
        ("triage", "emergency"): 0.95,
        # 兼容旧映射
        ("image", "radiology"): 0.95, ("image", "orthopedics"): 0.75,
        ("ecg", "cardiology"): 0.95,
        ("lab", "laboratory"): 0.90,
    }
    return match_map.get((data_type, dept), 0.30)


def _generate_suggestions(scores: Dict, dept: str) -> List[str]:
    """生成针对性改进建议"""
    suggestions = []
    for dim, score in scores.items():
        if score < 75:
            cn_map = {
                "completeness": f"完整性不足({score:.0f}分)：建议补充缺失字段，特别是关键字段如诊断结果、用药记录",
                "accuracy": f"准确性偏低({score:.0f}分)：建议增加交叉验证机制，引入二级审核",
                "timeliness": f"时效性不足({score:.0f}分)：建议缩短数据采集到入库的周期，设置更新提醒",
                "compliance": f"合规性风险({score:.0f}分)：建议补充知情同意书、脱敏处理记录",
            }
            suggestions.append(cn_map.get(dim, f"{dim}偏低({score:.0f}分)"))
    if not suggestions:
        suggestions.append("各维度质量良好，建议保持当前数据管理规范")
    return suggestions


# ============================================================
# MCP 工具定义 (7个)
# ============================================================
@mcp.tool()
def get_dataset_stats() -> Dict[str, Any]:
    """获取真实医疗数据集统计信息（5,000,000条Token的完整画像）。

    Returns:
        数据集统计：总行数、采样数、等级分布、科室分布、质量分范围
    """
    if not _HAS_REAL_DATA:
        return {
            "error": "真实数据未加载",
            "fallback": {
                "total_rows": 0,
                "note": "请运行 python data/loader.py 生成采样文件",
            },
        }

    stats = _LOADER.compute_stats()
    # 转换为可序列化的 dict
    return {
        "dataset_info": {
            "total_rows": stats["total_rows"],
            "sample_size": stats["sample_size"],
            "data_source": "北数所A级/B级医疗Token数据集",
            "fields": ["token_id", "domain", "category", "data_type", "entity_id",
                       "data_quality_score", "token_level", "completeness",
                       "accuracy", "timeliness", "compliance_score", "created_at"],
        },
        "level_distribution": stats["level_distribution"],
        "category_distribution": stats["category_distribution"],
        "data_type_distribution": stats["data_type_distribution"],
        "quality_stats": stats["quality_stats"],
        "by_department": stats["by_department"],
        "loaded_at": datetime.now().isoformat(),
    }


@mcp.tool()
def assess_data_quality(
    records: List[Dict[str, Any]],
    department: Optional[str] = None,
) -> Dict[str, Any]:
    """评估医疗数据质量：完整性、准确性、合规性、时效性，返回综合质量分和等级。

    Args:
        records: 医疗数据记录列表，每条记录应包含 completeness/accuracy/timeliness/compliance (0-100) 和 data_type
        department: 指定科室（可选，不指定则自动分类）

    Returns:
        质量评估结果，包含综合分、等级、各维度得分、改进建议
    """
    if not records:
        return {"error": "记录列表不能为空"}

    results = []
    for i, record in enumerate(records):
        scores = {}
        for dim in QUALITY_DIMENSIONS:
            val = record.get(dim, record.get(f"{dim}_score",
                       record.get("compliance_score" if dim == "compliance" else dim, 0)))
            scores[dim] = float(val) if val is not None else 0.0

        weights = {"completeness": 0.30, "accuracy": 0.35,
                    "timeliness": 0.15, "compliance": 0.20}
        overall = sum(scores[d] * weights[d] for d in QUALITY_DIMENSIONS)
        dept = department or _infer_department(record)
        level = _grade_level(overall)
        suggestions = _generate_suggestions(scores, dept)

        results.append({
            "record_index": i,
            "department": dept,
            "department_cn": DEPARTMENTS.get(dept, {}).get("name_cn", "未知"),
            "quality_score": round(overall, 2),
            "quality_level": level,
            "dimension_scores": {k: round(v, 2) for k, v in scores.items()},
            "suggestions": suggestions,
        })

    avg_score = sum(r["quality_score"] for r in results) / len(results)
    level_dist = {}
    dept_dist = {}
    for r in results:
        level_dist[r["quality_level"]] = level_dist.get(r["quality_level"], 0) + 1
        dept_dist[r["department"]] = dept_dist.get(r["department"], 0) + 1

    return {
        "total_records": len(records),
        "average_quality_score": round(avg_score, 2),
        "level_distribution": level_dist,
        "department_distribution": dept_dist,
        "details": results,
        "assessed_at": datetime.now().isoformat(),
    }


@mcp.tool()
def classify_department(record: Dict[str, Any]) -> Dict[str, Any]:
    """自动识别医疗数据所属科室（8大科室多分类）。

    Args:
        record: 医疗数据记录，包含 data_type, category 等字段

    Returns:
        科室分类结果，包含主科室、置信度、备选科室
    """
    dept = _infer_department(record)
    data_type = record.get("data_type", "text")
    confidence = min(_match_score(data_type, dept), 0.99)

    alternatives = []
    for d, info in DEPARTMENTS.items():
        if d != dept:
            alt_score = _match_score(data_type, d)
            if alt_score > 0.3:
                alternatives.append({
                    "department": d,
                    "name_cn": info["name_cn"],
                    "score": round(alt_score, 2),
                })
    alternatives.sort(key=lambda x: x["score"], reverse=True)

    return {
        "primary_department": dept,
        "department_cn": DEPARTMENTS.get(dept, {}).get("name_cn", "未知"),
        "confidence": round(confidence, 2),
        "alternatives": alternatives[:3],
        "data_type": data_type,
        "data_type_cn": DATA_TYPES.get(data_type, {}).get("name_cn", "未知"),
    }


@mcp.tool()
def grade_data_level(quality_score: float) -> Dict[str, Any]:
    """评定医疗数据等级（A级/B级/C级/D级），给出推荐用途和定价系数。

    Args:
        quality_score: 综合质量分 (0-100)

    Returns:
        数据等级及推荐用途
    """
    level = _grade_level(quality_score)
    level_info = {
        "A": {"name": "A级", "desc": "高质量数据，可直接用于AI模型训练",
              "recommended_use": "模型训练、临床决策支持", "price_multiplier": 1.50},
        "B": {"name": "B级", "desc": "良好质量数据，经清洗后可用于训练",
              "recommended_use": "模型预训练、数据分析", "price_multiplier": 1.00},
        "C": {"name": "C级", "desc": "基础质量数据，需人工审核后使用",
              "recommended_use": "统计分析、趋势研究", "price_multiplier": 0.60},
        "D": {"name": "D级", "desc": "质量不达标，不建议用于AI训练",
              "recommended_use": "仅限内部参考", "price_multiplier": 0.30},
    }
    info = level_info[level]
    return {
        "quality_score": quality_score,
        "level": level,
        "level_name": info["name"],
        "description": info["desc"],
        "recommended_use": info["recommended_use"],
        "price_multiplier": info["price_multiplier"],
    }


@mcp.tool()
def generate_quality_report(
    records: List[Dict[str, Any]],
    dataset_name: str = "未命名数据集",
) -> Dict[str, Any]:
    """生成完整的医疗数据质量报告（含科室分布、维度分析、改进建议）。

    Args:
        records: 医疗数据记录列表
        dataset_name: 数据集名称

    Returns:
        完整的质量评估报告
    """
    assessment = assess_data_quality(records, None)

    dept_stats = {}
    for r in assessment["details"]:
        d = r["department"]
        if d not in dept_stats:
            dept_stats[d] = {"name_cn": r["department_cn"], "count": 0,
                             "scores": [], "levels": {}}
        dept_stats[d]["count"] += 1
        dept_stats[d]["scores"].append(r["quality_score"])
        lvl = r["quality_level"]
        dept_stats[d]["levels"][lvl] = dept_stats[d]["levels"].get(lvl, 0) + 1

    for d, s in dept_stats.items():
        s["avg_score"] = round(sum(s["scores"]) / len(s["scores"]), 2)
        del s["scores"]

    dim_analysis = {}
    for dim in QUALITY_DIMENSIONS:
        scores = [r["dimension_scores"][dim] for r in assessment["details"]]
        dim_analysis[dim] = {
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "avg": round(sum(scores) / len(scores), 2),
            "weak": min(scores) < 75,
        }

    weak_dims = [d for d, v in dim_analysis.items() if v["weak"]]
    global_suggestions = []
    dim_cn = {"completeness": "完整性", "accuracy": "准确性",
              "timeliness": "时效性", "compliance": "合规性"}
    for dim in weak_dims:
        global_suggestions.append(f"全局{dim_cn.get(dim, dim)}偏低，建议优先改善数据采集流程中的{dim_cn.get(dim, dim)}环节")
    for dept, stats in dept_stats.items():
        if stats["avg_score"] < 75:
            global_suggestions.append(f"{stats['name_cn']}数据质量偏低({stats['avg_score']:.1f}分)，建议加强该科室数据录入培训")

    return {
        "dataset_name": dataset_name,
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_records": assessment["total_records"],
            "average_quality": assessment["average_quality_score"],
            "overall_level": _grade_level(assessment["average_quality_score"]),
            "level_distribution": assessment["level_distribution"],
        },
        "department_analysis": dept_stats,
        "dimension_analysis": dim_analysis,
        "global_suggestions": global_suggestions,
        "details": assessment["details"],
    }


@mcp.tool()
def search_similar_data(
    quality_profile: Dict[str, float],
    department: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """检索与给定质量画像相似的真实历史数据（基于5,000,000条Token数据集）。

    Args:
        quality_profile: 质量画像 {completeness, accuracy, timeliness, compliance}
        department: 限定科室（可选）
        top_k: 返回前K条（默认10，最大100）

    Returns:
        相似数据列表及匹配度（基于真实Token_id）
    """
    top_k = max(1, min(top_k, 100))

    if _HAS_REAL_DATA:
        results = _LOADER.search_similar(quality_profile, department, top_k)
        return {
            "data_source": "真实数据集 (5,000,000条Token采样)",
            "query_profile": quality_profile,
            "department_filter": department,
            "total_found": len(results),
            "top_results": results,
            "searched_at": datetime.now().isoformat(),
        }

    # 模拟回退
    results = []
    depts = [department] if department else list(DEPARTMENTS.keys())
    for d in depts:
        for i in range(5):
            base = 80 + (hash(d + str(i)) % 20)
            candidate = [base + 5, base, base - 5, base + 3]
            target = [quality_profile.get(dim, 0) for dim in QUALITY_DIMENSIONS]
            distance = sum((a - b) ** 2 for a, b in zip(target, candidate)) ** 0.5
            similarity = max(0, 1 - distance / 100)
            results.append({
                "record_id": f"{d}_{i:04d}",
                "department": d,
                "similarity": round(similarity, 4),
                "quality_score": base,
                "level": _grade_level(base),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "data_source": "模拟数据 (真实数据未加载)",
        "query_profile": quality_profile,
        "department_filter": department,
        "total_found": len(results),
        "top_results": results[:top_k],
    }


@mcp.tool()
def sample_real_records(
    n: int = 10,
    department: Optional[str] = None,
    level: Optional[str] = None,
) -> Dict[str, Any]:
    """从真实数据集中随机采样记录（用于数据预览和测试）。

    Args:
        n: 采样数量（默认10，最大1000）
        department: 限定科室（可选）
        level: 限定等级 A/B（可选）

    Returns:
        真实Token记录列表
    """
    n = max(1, min(n, 1000))

    if not _HAS_REAL_DATA:
        return {
            "error": "真实数据未加载",
            "fallback": "请运行 python data/loader.py 生成采样文件",
        }

    records = _LOADER.sample_records(n=n, department=department, level=level)
    return {
        "data_source": "真实数据集采样",
        "filters": {"department": department, "level": level},
        "total_returned": len(records),
        "records": records,
        "sampled_at": datetime.now().isoformat(),
    }


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    # 启动时预加载真实数据
    if _HAS_REAL_DATA:
        print("[Server] 预加载真实数据...", file=sys.stderr)
        _LOADER.load_sample()
        _LOADER.compute_stats()
        print("[Server] 数据加载完成，启动 MCP Server...", file=sys.stderr)
    else:
        print("[Server] 真实数据未加载，运行在模拟模式", file=sys.stderr)

    mcp.run(transport="stdio")
