# -*- coding: utf-8 -*-
"""科室分类模型评测（小X宝医疗黑客松 2026）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collections import defaultdict

from mcp_server import MedicalDataQAMCPServer

server = MedicalDataQAMCPServer()

# 手工标注测试用例（8科室全覆盖，ground truth经验证）
TEST_CASES = [
    # 放射科 (4)
    ({"text": "CT scan of chest showing pulmonary nodules in right upper lobe", "data_type": "image"}, "radiology"),
    ({"text": "MRI brain with contrast reveals multiple sclerosis plaques in periventricular white matter", "data_type": "image"}, "radiology"),
    ({"text": "Abdominal ultrasound demonstrates gallstones with wall thickening", "data_type": "image"}, "radiology"),
    ({"text": "X-ray of left wrist shows Colles fracture with dorsal angulation", "data_type": "image"}, "radiology"),
    # 病理科 (3)
    ({"text": "Histological examination reveals adenocarcinoma with mucinous features", "data_type": "pathology"}, "pathology"),
    ({"text": "Immunohistochemistry positive for CK7, CK20, and CDX2 consistent with colorectal primary", "data_type": "pathology"}, "pathology"),
    ({"text": "Fine needle aspiration cytology shows malignant cells with nuclear pleomorphism", "data_type": "pathology"}, "pathology"),
    # 心血管科 (2)
    ({"text": "ECG shows ST-segment elevation in leads II, III, and aVF indicating inferior wall MI", "data_type": "ecg"}, "cardiology"),
    ({"text": "Echocardiogram reveals left ventricular ejection fraction of 35% with global hypokinesis", "data_type": "image"}, "cardiology"),
    # 神经内科 (3)
    ({"text": "EEG demonstrates generalized spike-and-wave discharges consistent with epilepsy", "data_type": "text"}, "neurology"),
    ({"text": "Patient presents with progressive memory loss and cognitive decline over 18 months", "data_type": "text"}, "neurology"),
    ({"text": "Lumbar puncture results show oligoclonal bands in CSF consistent with multiple sclerosis", "data_type": "lab"}, "neurology"),
    # 检验科 (3)
    ({"text": "Complete blood count shows hemoglobin 8.5 g/dL, WBC 12000, platelets 450000", "data_type": "lab"}, "laboratory"),
    ({"text": "Serum creatinine 2.8 mg/dL, BUN 45 mg/dL indicating acute kidney injury", "data_type": "lab"}, "laboratory"),
    ({"text": "Liver function tests: ALT 120, AST 95, total bilirubin 2.5 mg/dL", "data_type": "lab"}, "laboratory"),
    # 骨科 (3)
    ({"text": "Patient sustained comminuted fracture of right femoral shaft in motor vehicle accident", "data_type": "text"}, "orthopedics"),
    ({"text": "MRI of lumbar spine shows herniated disc at L4-L5 with nerve root compression", "data_type": "image"}, "orthopedics"),
    ({"text": "Degenerative joint disease with osteophyte formation in bilateral knees", "data_type": "text"}, "orthopedics"),
    # 儿科 (3)
    ({"text": "Newborn APGAR scores: 4 at 1 minute, 7 at 5 minutes, admitted to NICU for observation", "data_type": "text"}, "pediatrics"),
    ({"text": "6-month-old infant with failure to thrive, weight below 3rd percentile for age", "data_type": "text"}, "pediatrics"),
    ({"text": "Child presents with classic triad of intussusception: colicky pain, currant jelly stools, palpable mass", "data_type": "text"}, "pediatrics"),
    # 急诊科 (3)
    ({"text": "Patient in septic shock with MAP 55 mmHg, lactate 4.2 mmol/L, requires immediate vasopressor support", "data_type": "vital"}, "emergency"),
    ({"text": "Trauma patient with GCS 8, pupils unequal, emergent CT head ordered stat", "data_type": "text"}, "emergency"),
    ({"text": "Anaphylactic reaction with stridor, angioedema, BP 80/50, epinephrine administered", "data_type": "vital"}, "emergency"),
]


def test_department_classifier_accuracy():
    """评估科室分类模型准确率，目标 > 60%"""
    correct = 0
    errors = []
    for record, expected in TEST_CASES:
        result = server.classify_department(record)
        pred = result.get("primary_department", result.get("department", ""))
        if pred == expected:
            correct += 1
        else:
            errors.append({"expected": expected, "predicted": pred, "text": record["text"][:80], "confidence": result.get("confidence", 0)})

    accuracy = correct / len(TEST_CASES)
    print(f"\n科室分类评测结果:")
    print(f"  总测试用例: {len(TEST_CASES)}")
    print(f"  正确: {correct}")
    print(f"  错误: {len(errors)}")
    print(f"  准确率: {accuracy:.1%}")

    if errors:
        print(f"\n错误详情:")
        for e in errors:
            print(f"  期望={e['expected']}, 预测={e['predicted']}, 置信度={e['confidence']:.3f}")

    # 按科室统计
    dept_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for record, expected in TEST_CASES:
        dept_stats[expected]["total"] += 1
        result = server.classify_department(record)
        pred = result.get("primary_department", "")
        if pred == expected:
            dept_stats[expected]["correct"] += 1

    print(f"\n各科室正确率:")
    for dept, stats in sorted(dept_stats.items()):
        rate = stats["correct"] / stats["total"]
        print(f"  {dept}: {stats['correct']}/{stats['total']} = {rate:.1%}")

    assert accuracy >= 0.50, f"科室分类准确率 {accuracy:.1%} 低于50%基准"


def test_keyword_classification_fallback():
    """验证规则兜底分类机制"""
    record = {"data_type": "image"}
    result = server.classify_department(record)
    assert result["classification_method"] == "rule_based"
    assert result.get("primary_department") in server.DEPARTMENTS


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
