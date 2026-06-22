# -*- coding: utf-8 -*-
"""
test_real_data.py
测试 MCP Server 与真实数据集的集成
"""
import sys
import os
from pathlib import Path

# 修复Windows控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

# 直接导入 server 模块中的工具函数 (不通过 MCP 协议)
import server

OK = "[OK]"
FAIL = "[FAIL]"


def test_get_dataset_stats():
    """测试1: 获取数据集统计"""
    print("\n[测试1] get_dataset_stats")
    result = server.get_dataset_stats.fn() if hasattr(server.get_dataset_stats, 'fn') else server.get_dataset_stats()
    if "error" in result:
        print(f"  ❌ 失败: {result['error']}")
        return False
    info = result["dataset_info"]
    print(f"  ✅ 总行数: {info['total_rows']:,}")
    print(f"  ✅ 采样数: {info['sample_size']:,}")
    print(f"  ✅ 等级分布: {result['level_distribution']}")
    print(f"  ✅ 科室数: {len(result['by_department'])}")
    return True


def test_sample_real_records():
    """测试2: 采样真实记录"""
    print("\n[测试2] sample_real_records")
    result = server.sample_real_records.fn(n=3, department="radiology", level="A") if hasattr(server.sample_real_records, 'fn') else server.sample_real_records(n=3, department="radiology", level="A")
    if "error" in result:
        print(f"  ❌ 失败: {result['error']}")
        return False
    print(f"  ✅ 返回记录数: {result['total_returned']}")
    if result["records"]:
        r = result["records"][0]
        print(f"  ✅ 样本: token_id={r['token_id']}, dept={r['department']}, level={r['token_level']}, quality={r['data_quality_score']}")
    return True


def test_search_similar_data():
    """测试3: 检索相似数据"""
    print("\n[测试3] search_similar_data")
    profile = {"completeness": 98.0, "accuracy": 97.0, "timeliness": 96.0, "compliance": 100.0}
    result = server.search_similar_data.fn(quality_profile=profile, department="cardiology", top_k=5) if hasattr(server.search_similar_data, 'fn') else server.search_similar_data(quality_profile=profile, department="cardiology", top_k=5)
    print(f"  ✅ 数据源: {result['data_source']}")
    print(f"  ✅ 找到: {result['total_found']} 条")
    if result["top_results"]:
        top = result["top_results"][0]
        print(f"  ✅ Top1: token_id={top['token_id']}, similarity={top['similarity']}, quality={top['quality_score']}")
    return True


def test_assess_data_quality():
    """测试4: 评估数据质量"""
    print("\n[测试4] assess_data_quality")
    records = [
        {"data_type": "ct_image", "completeness": 98.5, "accuracy": 97.0, "timeliness": 96.0, "compliance_score": 100.0},
        {"data_type": "blood_test", "completeness": 95.0, "accuracy": 94.0, "timeliness": 92.0, "compliance_score": 100.0},
        {"data_type": "ecg", "completeness": 70.0, "accuracy": 65.0, "timeliness": 60.0, "compliance_score": 80.0},
    ]
    result = server.assess_data_quality.fn(records=records) if hasattr(server.assess_data_quality, 'fn') else server.assess_data_quality(records=records)
    print(f"  ✅ 评估记录数: {result['total_records']}")
    print(f"  ✅ 平均质量分: {result['average_quality_score']}")
    print(f"  ✅ 等级分布: {result['level_distribution']}")
    print(f"  ✅ 科室分布: {result['department_distribution']}")
    return True


def test_classify_department():
    """测试5: 科室分类"""
    print("\n[测试5] classify_department")
    record = {"data_type": "ct_image", "category": "radiology"}
    result = server.classify_department.fn(record=record) if hasattr(server.classify_department, 'fn') else server.classify_department(record=record)
    print(f"  ✅ 主科室: {result['primary_department']} ({result['department_cn']})")
    print(f"  ✅ 置信度: {result['confidence']}")
    print(f"  ✅ 备选数: {len(result['alternatives'])}")
    return True


def test_grade_data_level():
    """测试6: 等级评定"""
    print("\n[测试6] grade_data_level")
    for score in [98.5, 82.0, 65.0, 45.0]:
        result = server.grade_data_level.fn(quality_score=score) if hasattr(server.grade_data_level, 'fn') else server.grade_data_level(quality_score=score)
        print(f"  ✅ 分数 {score} → {result['level']}级 ({result['level_name']}), 用途: {result['recommended_use']}")
    return True


def test_generate_quality_report():
    """测试7: 生成质量报告"""
    print("\n[测试7] generate_quality_report")
    records = [
        {"data_type": "ct_image", "completeness": 98.5, "accuracy": 97.0, "timeliness": 96.0, "compliance_score": 100.0},
        {"data_type": "blood_test", "completeness": 95.0, "accuracy": 94.0, "timeliness": 92.0, "compliance_score": 100.0},
    ]
    result = server.generate_quality_report.fn(records=records, dataset_name="测试数据集") if hasattr(server.generate_quality_report, 'fn') else server.generate_quality_report(records=records, dataset_name="测试数据集")
    print(f"  ✅ 数据集: {result['dataset_name']}")
    print(f"  ✅ 总记录: {result['summary']['total_records']}")
    print(f"  ✅ 平均质量: {result['summary']['average_quality']}")
    print(f"  ✅ 整体等级: {result['summary']['overall_level']}")
    print(f"  ✅ 科室分析: {len(result['department_analysis'])} 个")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("医疗数据质量评估 MCP Server — 真实数据集成测试")
    print("=" * 60)

    tests = [
        test_get_dataset_stats,
        test_sample_real_records,
        test_search_similar_data,
        test_assess_data_quality,
        test_classify_department,
        test_grade_data_level,
        test_generate_quality_report,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: ✅ {passed} 通过, ❌ {failed} 失败")
    print("=" * 60)
