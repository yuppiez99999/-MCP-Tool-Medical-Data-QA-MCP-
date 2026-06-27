# -*- coding: utf-8 -*-
"""
推理演示脚本 — 医疗 AI 模型系统
===============================
使用已训练的两个模型进行推理:
1. HealthcareTokenClassifier (PyTorch 多任务) — Token等级·质量分·科室
2. DepartmentClassifier (TF-IDF + LR) — 基于文献的科室分类
3. MCP Server 8工具完整演示
"""
import os
import sys
import json
import time
import pickle
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import torch

from data.loader import DataLoader
from data.preprocessor import HealthcareTokenPreprocessor
from models.classifier import HealthcareTokenClassifier
from mcp_server import MedicalDataQAMCPServer


DEPT_NAMES = {
    "radiology": "放射科", "pathology": "病理科", "neurology": "神经内科",
    "cardiovascular": "心血管科", "laboratory": "检验科", "orthopedics": "骨科",
    "pediatrics": "儿科", "emergency": "急诊科",
}
LEVEL_NAMES = {1: "A级", 0: "B级"}


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ================================================================
# Part 1: PyTorch 多任务分类器推理
# ================================================================
def inference_pytorch():
    print_section("PyTorch 多任务分类器推理")

    # 加载数据 & 模型
    loader = DataLoader.instance()
    df = loader.load_sample()
    categories = sorted(df["category"].unique().tolist())
    data_types = sorted(df["data_type"].unique().tolist())

    preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
    X, y_true = preprocessor.fit_transform(df)

    model_path = os.path.join(BASE_DIR, "outputs", "healthcare_token_classifier.pt")
    if not os.path.exists(model_path):
        print("[ERROR] 模型文件不存在:", model_path)
        return None
    model = HealthcareTokenClassifier.load(model_path)
    print(f"模型加载: {os.path.basename(model_path)}")
    print(f"  输入维度: {model.input_dim}, 科室数: {model.num_categories}")
    print(f"  设备: {model.device}")

    # 批量推理
    t0 = time.time()
    preds = model.predict(X[:5000], batch_size=2048)
    elapsed = time.time() - t0
    print(f"\n推理性能: 5,000 条 / {elapsed:.3f}s = {5000/elapsed:.0f} 条/秒")

    # Token等级
    level_acc = (preds["level_pred"] == y_true["level"][:5000]).mean() * 100
    level_a_ratio = preds["level_pred"].mean() * 100
    print(f"\nToken 等级分类:")
    print(f"  准确率: {level_acc:.1f}%")
    print(f"  A级占比(预测): {level_a_ratio:.1f}%")

    # 质量分
    q_mae = np.abs(preds["quality_pred"] - y_true["quality"][:5000]).mean()
    q_pred_mean = preds["quality_pred"].mean()
    q_true_mean = y_true["quality"][:5000].mean()
    print(f"\n质量分回归:")
    print(f"  MAE: {q_mae:.2f}")
    print(f"  预测均值: {q_pred_mean:.1f}  (真实均值: {q_true_mean:.1f})")

    # 按科室展示质量分
    print(f"\n各科室质量分预测:")
    for i, cat in enumerate(categories):
        mask = preds["category_pred"] == i
        if mask.sum() > 0:
            q_mean = preds["quality_pred"][mask].mean()
            q_true = y_true["quality"][:5000][mask].mean()
            print(f"  {DEPT_NAMES.get(cat, cat):8s}  预测:{q_mean:5.1f}  真实:{q_true:5.1f}  (n={mask.sum()})")

    # 单条Token详细输出
    print(f"\n单条Token推理详情 (前10条):")
    print(f"  {'序号':<4} {'科室':<8} {'真实等级':<8} {'预测等级':<8} {'置信度':<8} {'质量分':<8}")
    print(f"  {'-'*50}")
    for i in range(min(10, len(preds["level_pred"]))):
        true_lvl = "A级" if y_true["level"][i] == 1 else "B级"
        pred_lvl = "A级" if preds["level_pred"][i] == 1 else "B级"
        lvl_conf = preds["level_proba"][i][preds["level_pred"][i]] * 100
        true_cat = categories[y_true["category"][i]]
        q_val = preds["quality_pred"][i]
        cat_cn = DEPT_NAMES.get(true_cat, true_cat)
        print(f"  {i+1:<4} {cat_cn:<8} {true_lvl:<8} {pred_lvl:<8} {lvl_conf:5.1f}%  {q_val:6.1f}")

    return preds, y_true, categories


# ================================================================
# Part 2: TF-IDF 科室分类器推理
# ================================================================
def inference_department():
    print_section("TF-IDF 科室分类器推理 (文献训练)")

    pkl_path = os.path.join(BASE_DIR, "outputs", "department_classifier.pkl")
    if not os.path.exists(pkl_path):
        print("[ERROR] 科室分类器不存在:", pkl_path)
        return None

    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    pipeline = model_data["pipeline"]
    label_names = model_data["label_names"]
    print(f"模型加载: {os.path.basename(pkl_path)}")
    print(f"  类型: {model_data['model_type']}")
    print(f"  训练样本: {model_data.get('training_samples', 'N/A')}")

    # 各科室关键词
    if "top_keywords" in model_data:
        print(f"\n各科室Top关键词:")
        for dept, words in model_data["top_keywords"].items():
            cn = DEPT_NAMES.get(dept, dept)
            top5 = [w for w, _ in words[:5]]
            print(f"  {cn}: {', '.join(top5)}")

    # 自定义推理用例
    test_texts = [
        ("Chest CT shows bilateral ground-glass opacities suggestive of viral pneumonia", "未知"),
        ("Complete blood count reveals leukocytosis with neutrophilia", "未知"),
        ("Patient presents with sudden onset right-sided weakness and aphasia", "未知"),
        ("Coronary angiography demonstrates 90% stenosis in the left anterior descending artery", "未知"),
        ("Histopathology of breast biopsy shows invasive ductal carcinoma with lymphovascular invasion", "未知"),
        ("新生儿出生体重偏低，Apgar评分为6分，转入NICU观察", "未知"),
        ("Bone marrow aspirate shows hypercellularity with 25% blasts consistent with myelodysplasia", "未知"),
        ("急诊送来一名车祸重伤患者，血压70/40mmHg，神志不清", "未知"),
    ]

    print(f"\n自定义文本科室推理:")
    print(f"  {'文本摘要':<60} {'预测科室':<10} {'置信度':<8}")
    print(f"  {'-'*80}")
    for text, _ in test_texts[:8]:
        pred = pipeline.predict([text])[0]
        probs = pipeline.predict_proba([text])[0]
        conf = probs.max()
        cn = DEPT_NAMES.get(pred, pred)
        summary = text[:55] + "..." if len(text) > 55 else text
        print(f"  {summary:<60} {cn:<10} {conf:5.1%}")

    return pipeline, label_names


# ================================================================
# Part 3: MCP Server 8工具演示
# ================================================================
def inference_mcp_server():
    print_section("MCP Server 工具完整演示 (8个推理接口)")

    server = MedicalDataQAMCPServer()

    # 演示测试数据
    test_records = [
        {"completeness": 95, "accuracy": 92, "timeliness": 88, "compliance": 96,
         "data_type": "image", "department": "radiology",
         "text": "CT scan of the brain shows acute ischemic stroke"},

        {"completeness": 78, "accuracy": 85, "timeliness": 70, "compliance": 82,
         "data_type": "lab", "department": "laboratory",
         "text": "Complete blood count showing leukocytosis"},

        {"completeness": 60, "accuracy": 65, "timeliness": 55, "compliance": 70,
         "data_type": "text", "department": "pediatrics",
         "text": "Newborn screening test positive for phenylketonuria"},

        {"completeness": 88, "accuracy": 91, "timeliness": 85, "compliance": 93,
         "data_type": "ecg", "department": "cardiovascular",
         "text": "ECG demonstrating ST-elevation myocardial infarction in anterior leads"},

        {"completeness": 55, "accuracy": 58, "timeliness": 45, "compliance": 62,
         "data_type": "pathology", "department": "pathology",
         "text": "Fine needle aspiration cytology of thyroid nodule revealing papillary carcinoma"},
    ]

    # --- Tool 1: assess_data_quality ---
    print("\n[Tool 1] 数据质量评估:")
    result = server.assess_data_quality(test_records)
    print(f"  总记录: {result['total_records']}")
    print(f"  平均质量分: {result['average_quality_score']}")
    print(f"  等级分布: {result['level_distribution']}")
    print(f"  科室分布: {result['department_distribution']}")
    for d in result["details"][:3]:
        print(f"    [{d['department_cn']}] 质量分:{d['quality_score']} 等级:{d['quality_level']}")
        for s in d["suggestions"]:
            print(f"      → {s}")

    # --- Tool 2: classify_department ---
    print("\n[Tool 2] 科室自动分类 (ML模型):")
    for rec in test_records:
        r = server.classify_department(rec)
        method = "ML" if r["classification_method"] == "ml_model" else "规则引擎"
        print(f"  {r['data_type_cn']:6s} → {r['department_cn']:8s} (置信度:{r['confidence']:.0%}, {method})")

    # --- Tool 3: grade_data_level ---
    print("\n[Tool 3] 数据等级评定:")
    for score in [93, 82, 67, 45]:
        r = server.grade_data_level(score)
        print(f"  质量分={score} → {r['level_name']} | 推荐用途: {r['recommended_use']} | 价格系数: {r['price_multiplier']}x")

    # --- Tool 4: generate_quality_report ---
    print("\n[Tool 4] 完整质量报告:")
    report = server.generate_quality_report(test_records, "推理演示数据集")
    print(f"  数据集: {report['dataset_name']}")
    print(f"  综合等级: {report['summary']['overall_level']}")
    for dim, info in report["dimension_analysis"].items():
        flag = " ★弱" if info["weak"] else ""
        print(f"    {dim}: 均值={info['avg']:.1f} 最差={info['min']:.1f}{flag}")
    for s in report["global_suggestions"]:
        print(f"    → {s}")

    # --- Tool 5: search_similar_data ---
    print("\n[Tool 5] 相似数据检索:")
    profile = {"completeness": 82, "accuracy": 78, "timeliness": 70, "compliance": 85}
    similar = server.search_similar_data(profile, top_k=3)
    print(f"  检索画像: {profile}")
    for r in similar["top_results"]:
        print(f"    相似度:{r['similarity']:.3f} | 科室:{r['department']} | 等级:{r['level']} | 质量分:{r['quality_score']}")

    # --- Tool 6: search_medical_evidence ---
    print("\n[Tool 6] KnowS 医学循证检索:")
    evidence = server.search_medical_evidence("diabetic retinopathy deep learning screening", "paper_en", max_results=3)
    if "error" in evidence:
        print(f"  [需要 KNOWS_API_KEY 环境变量]: {evidence.get('error', evidence.get('hint', ''))}")
    else:
        for ev in evidence.get("evidences", [])[:3]:
            print(f"  [{ev['journal']}] {ev['title'][:60]}...")
            print(f"    DOI: {ev['doi']} | IF: {ev.get('impact_factor', 'N/A')}")

    # --- Tool 7: assess_with_evidence ---
    print("\n[Tool 7] 质量评估+文献检索联动:")
    aew = server.assess_with_evidence(test_records[:2], source="paper_en", evidence_count=2)
    print(f"  平均质量分: {aew['quality_assessment']['average_quality_score']}")
    relation = aew.get("evidence_relation", {})
    print(f"  主科室: {relation.get('department_cn', 'N/A')}")
    print(f"  薄弱维度: {relation.get('weak_dimensions', [])}")

    # --- Tool 8: generate_evidence_based_report ---
    print("\n[Tool 8] 循证医学文献引用质量报告:")
    ebr = server.generate_evidence_based_report(test_records, "推理演示数据集", evidence_per_dimension=1)
    if "error" in ebr:
        print(f"  [需要 KNOWS_API_KEY]: {ebr['error']}")
    else:
        print(f"  综合等级: {ebr['summary']['overall_level']}")
        print(f"  主科室: {ebr['summary'].get('primary_department_cn', 'N/A')}")
        print(f"  薄弱维度: {ebr['summary'].get('weak_dimensions', [])}")


# ================================================================
# Part 4: 综合推理演示 — 真实数据端到端
# ================================================================
def inference_end_to_end():
    print_section("端到端推理: 50,000条真实Token批量评分")

    loader = DataLoader.instance()
    df = loader.load_sample()
    categories = sorted(df["category"].unique().tolist())
    data_types = sorted(df["data_type"].unique().tolist())

    preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
    X, _ = preprocessor.fit_transform(df)

    model_path = os.path.join(BASE_DIR, "outputs", "healthcare_token_classifier.pt")
    model = HealthcareTokenClassifier.load(model_path)

    # 全量分批推理
    t0 = time.time()
    batch_size = 4096
    all_levels = []
    all_qualities = []
    all_categories = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        preds = model.predict(X[start:end], batch_size=batch_size)
        all_levels.append(preds["level_pred"])
        all_qualities.append(preds["quality_pred"])
        all_categories.append(preds["category_pred"])

    levels = np.concatenate(all_levels)
    qualities = np.concatenate(all_qualities)
    cate = np.concatenate(all_categories)
    elapsed = time.time() - t0
    print(f"推理完成: {len(df):,} 条 / {elapsed:.2f}s = {len(df)/elapsed:,.0f} 条/秒")

    # 统计
    a_count = (levels == 1).sum()
    b_count = (levels == 0).sum()
    print(f"\n等级分布:")
    print(f"  A级 Token: {a_count:,} ({a_count/len(levels)*100:.1f}%)")
    print(f"  B级 Token: {b_count:,} ({b_count/len(levels)*100:.1f}%)")

    print(f"\n质量分分布:")
    print(f"  均值: {qualities.mean():.2f}")
    print(f"  标准差: {qualities.std():.2f}")
    print(f"  P25: {np.percentile(qualities, 25):.1f}")
    print(f"  P50: {np.percentile(qualities, 50):.1f}")
    print(f"  P75: {np.percentile(qualities, 75):.1f}")

    print(f"\n科室分布 (预测):")
    for i, cat in enumerate(categories):
        cnt = (cate == i).sum()
        cn = DEPT_NAMES.get(cat, cat)
        avg_q = qualities[cate == i].mean()
        a_pct = (levels[cate == i] == 1).mean() * 100
        bar = "█" * int(cnt / len(cate) * 50)
        print(f"  {cn:8s} {cnt:6,}条  A级率:{a_pct:4.1f}%  质量分均值:{avg_q:5.1f}  {bar}")


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("  医疗 AI 模型系统 — 模型推理演示")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # 1. PyTorch 多任务推理
    inference_pytorch()

    # 2. TF-IDF 科室分类推理
    inference_department()

    # 3. MCP Server 8工具
    inference_mcp_server()

    # 4. 端到端全量推理
    inference_end_to_end()

    print_section("推理完成")
    print("  模型1: outputs/healthcare_token_classifier.pt (PyTorch 多任务)")
    print("  模型2: outputs/department_classifier.pkl (TF-IDF + LR)")
    print("  MCP Server: 8 个推理工具可用")


if __name__ == "__main__":
    main()
