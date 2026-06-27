# -*- coding: utf-8 -*-
"""MCP Server 8大工具功能测试（小X宝医疗黑客松 2026）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server import MedicalDataQAMCPServer

server = MedicalDataQAMCPServer()


def test_assess_data_quality():
    """Tool 1: assess_data_quality — 4维度加权评分"""
    records = [
        {"token_id": "T-001", "completeness": 98, "accuracy": 95, "timeliness": 90, "compliance": 100},
        {"token_id": "T-002", "completeness": 85, "accuracy": 80, "timeliness": 70, "compliance": 90},
    ]
    result = server.assess_data_quality(records, department="radiology")
    assert result["tool"] == "assess_data_quality"
    assert result["total_records"] == 2
    assert "average_quality_score" in result
    assert 0 <= result["average_quality_score"] <= 100
    for d in result["details"]:
        assert len(d["dimension_scores"]) == 4
        assert d["quality_level"] in ("A", "B", "C", "D")


def test_classify_department_text():
    """Tool 2: classify_department — ML模型文本分类"""
    record = {"text": "CT scan of brain showing intracranial hemorrhage", "data_type": "image"}
    result = server.classify_department(record)
    assert result["tool"] == "classify_department"
    assert "primary_department" in result
    assert "department_cn" in result
    assert "confidence" in result
    assert result["classification_method"] in ("ml_model", "rule_based")


def test_classify_department_rule():
    """Tool 2: classify_department — 规则兜底分类"""
    record = {"data_type": "image"}
    result = server.classify_department(record)
    assert "primary_department" in result
    assert result["classification_method"] == "rule_based"


def test_grade_data_level():
    """Tool 3: grade_data_level — A/B/C/D四级评定+定价系数"""
    for score, expected in [(93, "A"), (82, "B"), (67, "C"), (45, "D")]:
        r = server.grade_data_level(score)
        assert r["level"] == expected, f"score={score} expected {expected}, got {r.get('level')}"
        assert "price_multiplier" in r
        assert "recommended_use" in r


def test_generate_quality_report():
    """Tool 4: generate_quality_report — 完整质量报告生成"""
    records = [
        {"token_id": "T-001", "category": "radiology", "completeness": 95, "accuracy": 90, "timeliness": 85, "compliance": 95},
        {"token_id": "T-002", "category": "pathology", "completeness": 88, "accuracy": 92, "timeliness": 80, "compliance": 90},
    ]
    result = server.generate_quality_report(records, dataset_name="测试数据集")
    assert result["tool"] == "generate_quality_report"
    assert "dataset_name" in result
    assert "summary" in result
    assert "dimension_analysis" in result
    assert "global_suggestions" in result


def test_search_similar_data():
    """Tool 5: search_similar_data — 余弦相似度检索"""
    profile = {"completeness": 85, "accuracy": 80, "timeliness": 75, "compliance": 90}
    result = server.search_similar_data(profile, top_k=5)
    assert "total_found" in result
    assert "top_results" in result
    assert isinstance(result["top_results"], list)


def test_search_medical_evidence_no_key():
    """Tool 6: search_medical_evidence — 无API Key优雅降级"""
    result = server.search_medical_evidence(query="diabetes screening", source="paper_en", max_results=3)
    assert isinstance(result, dict)
    assert result["tool"] == "search_medical_evidence"


def test_assess_with_evidence():
    """Tool 7: assess_with_evidence — 质量评估+文献联动"""
    records = [
        {"token_id": "T-001", "completeness": 95, "accuracy": 90, "timeliness": 85, "compliance": 100},
    ]
    result = server.assess_with_evidence(records, department="radiology")
    assert result["tool"] == "assess_with_evidence"
    assert "quality_assessment" in result
    assert "evidence_search" in result


def test_generate_evidence_based_report():
    """Tool 8: generate_evidence_based_report — 带文献引用的完整报告"""
    records = [
        {"token_id": "T-001", "completeness": 80, "accuracy": 70, "timeliness": 60, "compliance": 85},
    ]
    result = server.generate_evidence_based_report(records, department="radiology", dataset_name="测试")
    assert result["tool"] == "generate_evidence_based_report"
    assert "summary" in result
    assert "evidence_by_dimension" in result
    assert "all_references" in result


def test_medical_compliance_disclaimer():
    """验证 mcp_server.py 包含医疗合规声明"""
    mcp_path = Path(__file__).resolve().parent.parent / "mcp_server.py"
    content = mcp_path.read_text(encoding="utf-8")
    assert "不提供任何医疗诊断建议" in content, "mcp_server.py 缺少医疗合规声明"


if __name__ == "__main__":
    tests = [
        test_assess_data_quality, test_classify_department_text,
        test_classify_department_rule, test_grade_data_level,
        test_generate_quality_report, test_search_similar_data,
        test_search_medical_evidence_no_key, test_assess_with_evidence,
        test_generate_evidence_based_report, test_medical_compliance_disclaimer,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__} — {e}")
    print(f"\n{passed}/{len(tests)} TESTS PASSED")
