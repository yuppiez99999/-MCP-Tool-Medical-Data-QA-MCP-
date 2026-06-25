# -*- coding: utf-8 -*-
"""
医疗数据质量评估 MCP Server
===========================
小X宝医疗黑客松参赛作品 — 可复用的 MCP 扩展工具

功能：
  1. assess_data_quality   — 评估医疗数据质量（完整性/准确性/合规性/时效性）
  2. classify_department   — 自动识别医疗科室（8大科室多分类）
  3. grade_data_level      — 数据等级评定（A/B/C级）
  4. generate_quality_report — 生成完整质量报告+改进建议
  5. search_similar_data   — 检索相似质量画像的历史数据

数据基础：390万条医疗健康Token数据，覆盖8大科室
部署目标：ModelScope MCP Server / Space
开源协议：MIT
"""
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ============================================================
# MCP Server 框架（轻量实现，无需第三方MCP依赖）
# ============================================================
class MedicalDataQAMCPServer:
    """医疗数据质量评估 MCP Server"""

    # 8大科室定义（复用 config.yaml）
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

    # 数据类型权重
    DATA_TYPES = {
        "image":    {"name_cn": "影像", "weight": 1.40},
        "text":     {"name_cn": "文本", "weight": 1.00},
        "ecg":      {"name_cn": "心电", "weight": 1.30},
        "lab":      {"name_cn": "检验", "weight": 0.95},
        "pathology":{"name_cn": "病理", "weight": 1.35},
        "genetic":  {"name_cn": "基因", "weight": 1.50},
        "vital":    {"name_cn": "生命体征", "weight": 0.80},
        "other":    {"name_cn": "其他", "weight": 0.70},
    }

    # 质量评估维度
    QUALITY_DIMENSIONS = [
        "completeness",   # 完整性
        "accuracy",       # 准确性
        "timeliness",     # 时效性
        "compliance",     # 合规性
    ]

    # KnowS 医学循证 API 配置
    KNOWS_BASE_URL = "https://api.nullht.com/v1"
    KNOWS_SOURCES = {
        "paper_en":       {"label": "英文论文", "endpoint": "/evidences/ai_search_paper_en", "max": 40},
        "paper_cn":       {"label": "中文论文", "endpoint": "/evidences/ai_search_paper_cn", "max": 40},
        "meeting":        {"label": "会议论文", "endpoint": "/evidences/ai_search_meeting", "max": 5},
        "guide":          {"label": "临床指南", "endpoint": "/evidences/ai_search_guide", "max": 5},
        "trial":          {"label": "临床试验", "endpoint": "/evidences/ai_search_trial", "max": 5},
        "package_insert": {"label": "药品说明书", "endpoint": "/evidences/ai_search_package_insert", "max": 5},
    }

    def __init__(self):
        self.version = "1.1.0"
        self.name = "medical-data-qa-mcp"
        self.knows_api_key = os.environ.get("KNOWS_API_KEY", "")

    # ============================================================
    # Tool 1: assess_data_quality — 评估医疗数据质量
    # ============================================================
    def assess_data_quality(
        self,
        records: List[Dict[str, Any]],
        department: Optional[str] = None,
    ) -> Dict[str, Any]:
        """评估医疗数据质量

        Args:
            records: 医疗数据记录列表，每条记录应包含:
                - completeness (0-100): 完整性得分
                - accuracy (0-100): 准确性得分
                - timeliness (0-100): 时效性得分
                - compliance (0-100): 合规性得分
                - data_type (str): 数据类型 (image/text/ecg/lab/...)
            department: 指定科室（可选，不指定则自动分类）

        Returns:
            质量评估结果，包含综合分、等级、各维度得分、改进建议
        """
        if not records:
            return {"error": "记录列表不能为空"}

        results = []
        for i, record in enumerate(records):
            # 提取质量维度得分
            scores = {}
            for dim in self.QUALITY_DIMENSIONS:
                val = record.get(dim, record.get(f"{dim}_score", 0))
                scores[dim] = float(val) if val is not None else 0.0

            # 计算综合质量分（加权平均）
            weights = {"completeness": 0.30, "accuracy": 0.35,
                       "timeliness": 0.15, "compliance": 0.20}
            overall = sum(scores[d] * weights[d] for d in self.QUALITY_DIMENSIONS)

            # 科室分类（如未指定则基于数据类型推断）
            dept = department or self._infer_department(record)

            # 数据等级评定
            level = self._grade_level(overall)

            # 生成改进建议
            suggestions = self._generate_suggestions(scores, dept)

            results.append({
                "record_index": i,
                "department": dept,
                "department_cn": self.DEPARTMENTS.get(dept, {}).get("name_cn", "未知"),
                "quality_score": round(overall, 2),
                "quality_level": level,
                "dimension_scores": {k: round(v, 2) for k, v in scores.items()},
                "suggestions": suggestions,
            })

        # 汇总统计
        avg_score = sum(r["quality_score"] for r in results) / len(results)
        level_dist = {}
        dept_dist = {}
        for r in results:
            level_dist[r["quality_level"]] = level_dist.get(r["quality_level"], 0) + 1
            dept_dist[r["department"]] = dept_dist.get(r["department"], 0) + 1

        return {
            "tool": "assess_data_quality",
            "total_records": len(records),
            "average_quality_score": round(avg_score, 2),
            "level_distribution": level_dist,
            "department_distribution": dept_dist,
            "details": results,
            "assessed_at": datetime.now().isoformat(),
        }

    # ============================================================
    # Tool 2: classify_department — 自动识别医疗科室
    # ============================================================
    def classify_department(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """自动识别医疗数据所属科室

        Args:
            record: 医疗数据记录，包含 data_type, category 等字段

        Returns:
            科室分类结果，包含主科室、置信度、备选科室
        """
        dept = self._infer_department(record)
        confidence = self._calc_confidence(record, dept)

        # 备选科室（基于数据类型的次优匹配）
        alternatives = []
        data_type = record.get("data_type", "text")
        for d, info in self.DEPARTMENTS.items():
            if d != dept:
                alt_score = self._match_score(data_type, d)
                if alt_score > 0.3:
                    alternatives.append({
                        "department": d,
                        "name_cn": info["name_cn"],
                        "score": round(alt_score, 2),
                    })
        alternatives.sort(key=lambda x: x["score"], reverse=True)

        return {
            "tool": "classify_department",
            "primary_department": dept,
            "department_cn": self.DEPARTMENTS.get(dept, {}).get("name_cn", "未知"),
            "confidence": round(confidence, 2),
            "alternatives": alternatives[:3],
            "data_type": data_type,
            "data_type_cn": self.DATA_TYPES.get(data_type, {}).get("name_cn", "未知"),
        }

    # ============================================================
    # Tool 3: grade_data_level — 数据等级评定
    # ============================================================
    def grade_data_level(self, quality_score: float) -> Dict[str, Any]:
        """评定医疗数据等级

        Args:
            quality_score: 综合质量分 (0-100)

        Returns:
            数据等级（A级/B级/C级）及说明
        """
        level = self._grade_level(quality_score)
        level_info = {
            "A": {
                "name": "A级",
                "desc": "高质量数据，可直接用于AI模型训练",
                "min_score": 90,
                "recommended_use": "模型训练、临床决策支持",
                "price_multiplier": 1.50,
            },
            "B": {
                "name": "B级",
                "desc": "良好质量数据，经清洗后可用于训练",
                "min_score": 75,
                "recommended_use": "模型预训练、数据分析",
                "price_multiplier": 1.00,
            },
            "C": {
                "name": "C级",
                "desc": "基础质量数据，需人工审核后使用",
                "min_score": 60,
                "recommended_use": "统计分析、趋势研究",
                "price_multiplier": 0.60,
            },
            "D": {
                "name": "D级",
                "desc": "质量不达标，不建议用于AI训练",
                "min_score": 0,
                "recommended_use": "仅限内部参考",
                "price_multiplier": 0.30,
            },
        }
        info = level_info[level]
        return {
            "tool": "grade_data_level",
            "quality_score": quality_score,
            "level": level,
            "level_name": info["name"],
            "description": info["desc"],
            "recommended_use": info["recommended_use"],
            "price_multiplier": info["price_multiplier"],
        }

    # ============================================================
    # Tool 4: generate_quality_report — 生成完整质量报告
    # ============================================================
    def generate_quality_report(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str = "未命名数据集",
    ) -> Dict[str, Any]:
        """生成完整的医疗数据质量报告

        Args:
            records: 医疗数据记录列表
            dataset_name: 数据集名称

        Returns:
            完整的质量评估报告（含统计、分布、建议）
        """
        assessment = self.assess_data_quality(records)

        # 科室分布统计
        dept_stats = {}
        for r in assessment["details"]:
            d = r["department"]
            if d not in dept_stats:
                dept_stats[d] = {
                    "name_cn": r["department_cn"],
                    "count": 0,
                    "scores": [],
                    "levels": {},
                }
            dept_stats[d]["count"] += 1
            dept_stats[d]["scores"].append(r["quality_score"])
            lvl = r["quality_level"]
            dept_stats[d]["levels"][lvl] = dept_stats[d]["levels"].get(lvl, 0) + 1

        # 计算各科室平均分
        for d, s in dept_stats.items():
            s["avg_score"] = round(sum(s["scores"]) / len(s["scores"]), 2)
            del s["scores"]

        # 质量维度分析
        dim_analysis = {}
        for dim in self.QUALITY_DIMENSIONS:
            scores = [r["dimension_scores"][dim] for r in assessment["details"]]
            dim_analysis[dim] = {
                "min": round(min(scores), 2),
                "max": round(max(scores), 2),
                "avg": round(sum(scores) / len(scores), 2),
                "weak": min(scores) < 75,
            }

        # 全局改进建议
        weak_dims = [d for d, v in dim_analysis.items() if v["weak"]]
        global_suggestions = self._global_suggestions(weak_dims, dept_stats)

        return {
            "tool": "generate_quality_report",
            "dataset_name": dataset_name,
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_records": assessment["total_records"],
                "average_quality": assessment["average_quality_score"],
                "overall_level": self._grade_level(assessment["average_quality_score"]),
                "level_distribution": assessment["level_distribution"],
            },
            "department_analysis": dept_stats,
            "dimension_analysis": dim_analysis,
            "global_suggestions": global_suggestions,
            "details": assessment["details"],
        }

    # ============================================================
    # Tool 5: search_similar_data — 检索相似质量画像数据
    # ============================================================
    def search_similar_data(
        self,
        quality_profile: Dict[str, float],
        department: Optional[str] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """检索与给定质量画像相似的历史数据

        Args:
            quality_profile: 质量画像 {completeness, accuracy, timeliness, compliance}
            department: 限定科室（可选）
            top_k: 返回前K条

        Returns:
            相似数据列表及匹配度
        """
        # 模拟历史数据库（实际部署时替换为真实数据源）
        mock_history = self._load_sample_history(department)

        # 计算相似度（欧氏距离）
        results = []
        target = [quality_profile.get(d, 0) for d in self.QUALITY_DIMENSIONS]
        for record in mock_history:
            candidate = [record["dimension_scores"].get(d, 0) for d in self.QUALITY_DIMENSIONS]
            distance = sum((a - b) ** 2 for a, b in zip(target, candidate)) ** 0.5
            similarity = max(0, 1 - distance / 100)
            results.append({
                "record_id": record.get("token_id", "N/A"),
                "department": record.get("department", "unknown"),
                "similarity": round(similarity, 4),
                "quality_score": record.get("quality_score", 0),
                "level": record.get("level", "C"),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return {
            "tool": "search_similar_data",
            "query_profile": quality_profile,
            "department_filter": department,
            "total_found": len(results),
            "top_results": results[:top_k],
        }

    # ============================================================
    # Tool 6: search_medical_evidence — KnowS 医学循证检索
    # ============================================================
    def search_medical_evidence(
        self,
        query: str,
        source: str = "paper_en",
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """检索医学循证证据（KnowS API）

        Args:
            query: 检索关键词（支持中英文）
            source: 数据源 (paper_en/paper_cn/meeting/guide/trial/package_insert)
            max_results: 返回结果数量（不超过数据源最大值）

        Returns:
            检索结果，包含文献列表、元数据、分页信息
        """
        if not query or not query.strip():
            return {"error": "检索关键词不能为空"}

        if source not in self.KNOWS_SOURCES:
            return {
                "error": f"不支持的数据源: {source}",
                "available_sources": list(self.KNOWS_SOURCES.keys()),
            }

        if not HAS_REQUESTS:
            return {"error": "requests 库未安装，请先安装: pip install requests"}

        if not self.knows_api_key:
            return {
                "error": "KNOWS_API_KEY 未配置",
                "hint": "请设置环境变量 KNOWS_API_KEY 或在初始化时传入 api_key",
            }

        source_info = self.KNOWS_SOURCES[source]
        endpoint = source_info["endpoint"]
        url = f"{self.KNOWS_BASE_URL.rstrip('/')}{endpoint}"
        limit = min(max_results, source_info["max"])

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.knows_api_key}",
        }
        payload = {"query": query}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return {"error": "KnowS API 请求超时，请稍后重试"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"KnowS API 请求失败: HTTP {resp.status_code}", "details": str(e)}
        except requests.exceptions.RequestException as e:
            return {"error": f"网络请求异常: {e}"}
        except Exception as e:
            return {"error": f"解析响应失败: {e}"}

        evidences = data.get("evidences", [])
        question_id = data.get("question_id", "")

        formatted = []
        for ev in evidences[:limit]:
            formatted.append({
                "id": ev.get("id", ""),
                "title": ev.get("title", ""),
                "abstract": ev.get("abstract", ""),
                "journal": ev.get("journal", ""),
                "publish_date": ev.get("publish_date", ""),
                "authors": ev.get("authors", []),
                "doi": ev.get("doi", ""),
                "study_type": ev.get("study_type", ""),
                "impact_factor": ev.get("impact_factor", ""),
                "has_pdf": ev.get("has_pdf", False),
                "cas_division": ev.get("cas_journal_division", ""),
                "wos_quartile": ev.get("wos_jif_quartile", ""),
            })

        return {
            "tool": "search_medical_evidence",
            "source": source,
            "source_label": source_info["label"],
            "query": query,
            "question_id": question_id,
            "total_found": len(evidences),
            "returned": len(formatted),
            "evidences": formatted,
            "searched_at": datetime.now().isoformat(),
        }

    # ============================================================
    # Tool 7: assess_with_evidence — 质量评估 + 文献检索联动
    # ============================================================
    def assess_with_evidence(
        self,
        records: List[Dict[str, Any]],
        query: str = "",
        source: str = "paper_en",
        department: Optional[str] = None,
        evidence_count: int = 5,
    ) -> Dict[str, Any]:
        """评估数据质量并联动检索相关医学文献

        评估完成后，自动根据科室和数据类型生成检索词，
        检索相关领域最新文献，为数据质量改进提供循证依据。

        Args:
            records: 医疗数据记录列表
            query: 自定义检索词（可选，不填则自动生成）
            source: 文献数据源
            department: 指定科室（可选）
            evidence_count: 返回文献数量

        Returns:
            质量评估结果 + 相关医学循证文献
        """
        if not records:
            return {"error": "记录列表不能为空"}

        assessment = self.assess_data_quality(records, department)
        if "error" in assessment:
            return assessment

        avg_score = assessment["average_quality_score"]
        primary_dept = ""
        max_count = 0
        for dept, cnt in assessment["department_distribution"].items():
            if cnt > max_count:
                max_count = cnt
                primary_dept = dept

        dept_cn = self.DEPARTMENTS.get(primary_dept, {}).get("name_cn", "")

        if not query.strip():
            quality_concern = ""
            if avg_score < 75:
                quality_concern = "数据质量改进 医疗数据治理"
            elif avg_score < 90:
                quality_concern = "医疗数据质量管理"
            else:
                quality_concern = "高质量医疗数据 AI训练"

            if dept_cn:
                query = f"{dept_cn} {quality_concern}"
            else:
                query = f"医疗数据质量 {quality_concern}"

        evidence_result = self.search_medical_evidence(query, source, evidence_count)

        weak_dims = []
        dim_cn = {
            "completeness": "完整性",
            "accuracy": "准确性",
            "timeliness": "时效性",
            "compliance": "合规性",
        }
        dim_scores = {}
        for dim in self.QUALITY_DIMENSIONS:
            scores = [r["dimension_scores"][dim] for r in assessment["details"]]
            dim_scores[dim] = sum(scores) / len(scores)
            if dim_scores[dim] < 75:
                weak_dims.append(dim_cn[dim])

        evidence_relation = {
            "primary_department": primary_dept,
            "department_cn": dept_cn,
            "average_quality_score": round(avg_score, 2),
            "weak_dimensions": weak_dims,
            "auto_generated_query": not query.strip() or query != query,
            "search_query_used": query,
            "purpose": (
                f"为{dept_cn + ' ' if dept_cn else ''}数据质量改进提供循证医学依据，"
                f"重点关注{'、'.join(weak_dims) if weak_dims else '全维度质量提升'}"
            ),
        }

        return {
            "tool": "assess_with_evidence",
            "quality_assessment": assessment,
            "evidence_search": evidence_result,
            "evidence_relation": evidence_relation,
            "assessed_at": datetime.now().isoformat(),
        }

    # ============================================================
    # 内部辅助方法
    # ============================================================
    def _infer_department(self, record: Dict) -> str:
        """基于数据类型和内容推断科室"""
        data_type = record.get("data_type", "text")
        category = record.get("category", record.get("department", ""))

        # 如果明确指定了科室，直接返回
        if category and category in self.DEPARTMENTS:
            return category

        # 基于数据类型推断
        type_dept_map = {
            "image": "radiology",
            "ecg": "cardiology",
            "lab": "laboratory",
            "pathology": "pathology",
            "genetic": "neurology",
            "vital": "emergency",
        }
        return type_dept_map.get(data_type, "laboratory")

    def _grade_level(self, score: float) -> str:
        """根据质量分评定等级"""
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"

    def _calc_confidence(self, record: Dict, dept: str) -> float:
        """计算分类置信度"""
        data_type = record.get("data_type", "text")
        score = self._match_score(data_type, dept)
        return min(score, 0.99)

    def _match_score(self, data_type: str, dept: str) -> float:
        """数据类型与科室的匹配度"""
        match_map = {
            ("image", "radiology"): 0.95,
            ("image", "orthopedics"): 0.75,
            ("image", "pathology"): 0.70,
            ("ecg", "cardiology"): 0.95,
            ("lab", "laboratory"): 0.90,
            ("pathology", "pathology"): 0.95,
            ("genetic", "neurology"): 0.80,
            ("vital", "emergency"): 0.85,
            ("text", "pediatrics"): 0.60,
        }
        return match_map.get((data_type, dept), 0.30)

    def _generate_suggestions(self, scores: Dict, dept: str) -> List[str]:
        """生成针对性改进建议"""
        suggestions = []
        dim_cn = {
            "completeness": "完整性",
            "accuracy": "准确性",
            "timeliness": "时效性",
            "compliance": "合规性",
        }
        for dim, score in scores.items():
            if score < 75:
                if dim == "completeness":
                    suggestions.append(f"完整性不足({score:.0f}分)：建议补充缺失字段，特别是关键字段如诊断结果、用药记录")
                elif dim == "accuracy":
                    suggestions.append(f"准确性偏低({score:.0f}分)：建议增加交叉验证机制，引入二级审核")
                elif dim == "timeliness":
                    suggestions.append(f"时效性不足({score:.0f}分)：建议缩短数据采集到入库的周期，设置更新提醒")
                elif dim == "compliance":
                    suggestions.append(f"合规性风险({score:.0f}分)：建议补充知情同意书、脱敏处理记录")
        if not suggestions:
            suggestions.append("各维度质量良好，建议保持当前数据管理规范")
        return suggestions

    def _global_suggestions(self, weak_dims: List[str], dept_stats: Dict) -> List[str]:
        """生成全局改进建议"""
        suggestions = []
        dim_cn = {
            "completeness": "完整性",
            "accuracy": "准确性",
            "timeliness": "时效性",
            "compliance": "合规性",
        }
        for dim in weak_dims:
            cn = dim_cn.get(dim, dim)
            suggestions.append(f"全局{cn}偏低，建议优先改善数据采集流程中的{cn}环节")

        # 科室级建议
        for dept, stats in dept_stats.items():
            if stats["avg_score"] < 75:
                name = stats["name_cn"]
                suggestions.append(f"{name}数据质量偏低({stats['avg_score']:.1f}分)，建议加强该科室数据录入培训")
        return suggestions

    def _load_sample_history(self, department: Optional[str] = None) -> List[Dict]:
        """加载样本历史数据（演示用，实际部署替换为真实数据源）"""
        # 基于真实数据分布的模拟样本
        samples = []
        depts = [department] if department else list(self.DEPARTMENTS.keys())
        for d in depts:
            for i in range(5):
                base = 80 + (hash(d + str(i)) % 20)
                samples.append({
                    "token_id": f"{d}_{i:04d}",
                    "department": d,
                    "quality_score": base,
                    "level": self._grade_level(base),
                    "dimension_scores": {
                        "completeness": base + 5,
                        "accuracy": base,
                        "timeliness": base - 5,
                        "compliance": base + 3,
                    },
                })
        return samples

    # ============================================================
    # MCP 协议接口 — 工具列表
    # ============================================================
    def list_tools(self) -> List[Dict]:
        """返回可用工具列表（MCP协议）"""
        return [
            {
                "name": "assess_data_quality",
                "description": "评估医疗数据质量：完整性、准确性、合规性、时效性，返回综合质量分和等级",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "description": "医疗数据记录列表",
                            "items": {"type": "object"},
                        },
                        "department": {"type": "string", "description": "指定科室（可选）"},
                    },
                    "required": ["records"],
                },
            },
            {
                "name": "classify_department",
                "description": "自动识别医疗数据所属科室（8大科室多分类）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "record": {"type": "object", "description": "医疗数据记录"},
                    },
                    "required": ["record"],
                },
            },
            {
                "name": "grade_data_level",
                "description": "评定医疗数据等级（A级/B级/C级/D级），给出推荐用途",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "quality_score": {"type": "number", "description": "综合质量分(0-100)"},
                    },
                    "required": ["quality_score"],
                },
            },
            {
                "name": "generate_quality_report",
                "description": "生成完整的医疗数据质量报告（含科室分布、维度分析、改进建议）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "records": {"type": "array", "items": {"type": "object"}},
                        "dataset_name": {"type": "string"},
                    },
                    "required": ["records"],
                },
            },
            {
                "name": "search_similar_data",
                "description": "检索与给定质量画像相似的历史数据",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "quality_profile": {"type": "object", "description": "质量画像"},
                        "department": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["quality_profile"],
                },
            },
            {
                "name": "search_medical_evidence",
                "description": "KnowS医学循证检索：检索英文/中文论文、指南、临床试验、药品说明书等医学证据",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词（支持中英文）"},
                        "source": {
                            "type": "string",
                            "description": "数据源：paper_en/paper_cn/meeting/guide/trial/package_insert",
                            "default": "paper_en",
                        },
                        "max_results": {"type": "integer", "description": "返回结果数", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "assess_with_evidence",
                "description": "质量评估+文献检索联动：评估数据质量后自动检索相关医学循证文献，提供改进依据",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "records": {"type": "array", "items": {"type": "object"}, "description": "医疗数据记录列表"},
                        "query": {"type": "string", "description": "自定义检索词（可选，自动生成）"},
                        "source": {"type": "string", "description": "文献数据源", "default": "paper_en"},
                        "department": {"type": "string", "description": "指定科室（可选）"},
                        "evidence_count": {"type": "integer", "description": "返回文献数", "default": 5},
                    },
                    "required": ["records"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: Dict) -> Any:
        """调用工具（MCP协议入口）"""
        if name == "assess_data_quality":
            return self.assess_data_quality(**arguments)
        elif name == "classify_department":
            return self.classify_department(**arguments)
        elif name == "grade_data_level":
            return self.grade_data_level(**arguments)
        elif name == "generate_quality_report":
            return self.generate_quality_report(**arguments)
        elif name == "search_similar_data":
            return self.search_similar_data(**arguments)
        elif name == "search_medical_evidence":
            return self.search_medical_evidence(**arguments)
        elif name == "assess_with_evidence":
            return self.assess_with_evidence(**arguments)
        else:
            return {"error": f"未知工具: {name}"}


# ============================================================
# 快速测试 / Demo
# ============================================================
if __name__ == "__main__":
    server = MedicalDataQAMCPServer()

    # 列出工具
    print("=== 可用工具 ===")
    for tool in server.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    # 测试1: 评估数据质量
    print("\n=== 测试: 评估数据质量 ===")
    test_records = [
        {
            "completeness": 95, "accuracy": 92, "timeliness": 88, "compliance": 96,
            "data_type": "image", "department": "radiology",
        },
        {
            "completeness": 78, "accuracy": 85, "timeliness": 70, "compliance": 82,
            "data_type": "lab", "department": "laboratory",
        },
        {
            "completeness": 60, "accuracy": 65, "timeliness": 55, "compliance": 70,
            "data_type": "text", "department": "pediatrics",
        },
    ]
    result = server.assess_data_quality(test_records)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 测试2: 生成完整报告
    print("\n=== 测试: 生成质量报告 ===")
    report = server.generate_quality_report(test_records, "测试数据集")
    print(f"数据集: {report['dataset_name']}")
    print(f"总记录: {report['summary']['total_records']}")
    print(f"平均分: {report['summary']['average_quality']}")
    print(f"等级分布: {report['summary']['level_distribution']}")
    print(f"改进建议: {report['global_suggestions']}")
