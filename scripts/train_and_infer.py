# -*- coding: utf-8 -*-
"""
训练+推理一体化流程脚本
======================
两阶段流水线:
  Phase 1 — 重新训练两大模型:
    Model A: HealthcareTokenClassifier (PyTorch 多任务)
    Model B: DepartmentClassifier (TF-IDF + LR, KnowS 文献驱动)
  Phase 2 — 全面推理验证 + 质量报告

用法:
  cd 18-医疗AI模型系统
  python scripts/train_and_infer.py                    # 完整训练+推理
  python scripts/train_and_infer.py --skip-pytorch     # 仅训练科室分类器+推理
  python scripts/train_and_infer.py --skip-knows       # 仅训练PyTorch模型+推理
  python scripts/train_and_infer.py --infer-only       # 仅推理（跳过训练）
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 加载 .env ──
try:
    from dotenv import load_dotenv
    _env_path = BASE_DIR / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_LINES: list = []


def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    LOG_LINES.append(line)


def section(title: str):
    log("")
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


def save_report(report: dict, name: str) -> str:
    path = REPORT_DIR / f"{name}_{TIMESTAMP}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    log(f"  报告已保存: {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════
# Phase 1A: 训练 PyTorch 多任务分类器
# ══════════════════════════════════════════════════════════════
def train_pytorch_classifier(args) -> dict:
    section("Phase 1A — 训练 PyTorch 多任务分类器")

    from data.loader import DataLoader
    from data.preprocessor import HealthcareTokenPreprocessor
    from models.classifier import HealthcareTokenClassifier

    # 加载采样数据
    loader = DataLoader.instance()
    df_sample = loader.load_sample(force_rebuild=getattr(args, "force_rebuild", False))
    log(f"  数据集: {len(df_sample):,} 条 Token")

    categories = sorted(df_sample["category"].unique().tolist())
    data_types = sorted(df_sample["data_type"].unique().tolist())
    log(f"  科室 ({len(categories)}): {', '.join(categories)}")
    log(f"  数据类型 ({len(data_types)}): {', '.join(data_types)}")

    # 抽样
    if args.pytorch_sample > 0 and args.pytorch_sample < len(df_sample):
        df = df_sample.sample(n=args.pytorch_sample, random_state=42).reset_index(drop=True)
        log(f"  训练抽样: {len(df):,} 条")
    else:
        df = df_sample.reset_index(drop=True)

    # 特征工程
    log(f"  特征工程中...")
    preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
    df_clean = preprocessor.clean(df)
    X_all, y_all = preprocessor.fit_transform(df_clean)
    log(f"  特征维度: {X_all.shape[1]}, 样本: {X_all.shape[0]:,}")

    level_a = int(y_all["level"].sum())
    log(f"  等级分布: A={level_a:,} B={len(y_all['level']) - level_a:,}")

    # 划分
    indices = np.arange(len(X_all))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42, stratify=y_all["level"])
    X_train, X_val = X_all[idx_train], X_all[idx_val]
    y_train = {k: v[idx_train] for k, v in y_all.items()}
    y_val = {k: v[idx_val] for k, v in y_all.items()}
    log(f"  训练集: {len(X_train):,} / 验证集: {len(X_val):,}")

    # 训练
    log(f"  设备: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    log(f"  超参数: lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}, early_stop={args.early_stop}")
    t0 = time.time()

    model = HealthcareTokenClassifier(
        input_dim=X_train.shape[1],
        num_categories=len(categories),
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        learning_rate=args.lr,
    )

    history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop,
        verbose=True,
    )

    train_time = round(time.time() - t0, 1)

    # 保存
    model_path = str(OUTPUT_DIR / "healthcare_token_classifier.pt")
    model.save(model_path)

    # 最终指标
    final_metrics = {
        "val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
        "level_acc": history["level_acc"][-1] if history["level_acc"] else 0,
        "quality_mae": history["quality_mae"][-1] if history["quality_mae"] else 0,
        "category_acc": history["category_acc"][-1] if history["category_acc"] else 0,
        "train_time_sec": train_time,
        "model_path": model_path,
        "model_size_kb": round(Path(model_path).stat().st_size / 1024, 1),
        "epochs_trained": len(history["train_loss"]),
    }

    log(f"\n  训练完成 ({train_time}s)")
    log(f"  等级准确率: {final_metrics['level_acc']:.4f}")
    log(f"  质量分 MAE:  {final_metrics['quality_mae']:.4f}")
    log(f"  科室准确率: {final_metrics['category_acc']:.4f}")
    log(f"  模型已保存: {model_path} ({final_metrics['model_size_kb']} KB)")

    final_metrics["history"] = {
        "train_loss": [round(v, 4) for v in history["train_loss"]],
        "val_loss": [round(v, 4) for v in history.get("val_loss", [])],
    }
    save_report({"phase": "1A_pytorch", "metrics": final_metrics}, "pytorch_train")
    return final_metrics


# ══════════════════════════════════════════════════════════════
# Phase 1B: 训练 TF-IDF 科室分类器（KnowS 文献驱动）
# ══════════════════════════════════════════════════════════════
def train_department_classifier(args) -> dict:
    section("Phase 1B — 训练 KnowS 文献驱动科室分类器")

    # 动态加载科室分类器训练模块
    import importlib.util
    _train_dept_path = BASE_DIR / "scripts" / "train_department_classifier.py"
    spec = importlib.util.spec_from_file_location("train_department_classifier", _train_dept_path)
    _train_dept_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_train_dept_mod)
    DEPARTMENT_QUERIES = _train_dept_mod.DEPARTMENT_QUERIES
    DEPARTMENT_NAMES = _train_dept_mod.DEPARTMENT_NAMES
    fetch_literature = _train_dept_mod.fetch_literature
    train_classifier = _train_dept_mod.train_classifier
    test_classifier = _train_dept_mod.test_classifier
    from mcp_server import MedicalDataQAMCPServer

    server = MedicalDataQAMCPServer()

    # 检索
    log(f"  从 KnowS API 检索 {len(DEPARTMENT_QUERIES)} 个科室文献...")
    t0 = time.time()
    literature_data = fetch_literature(server, max_per_query=args.knows_per_query)
    fetch_time = round(time.time() - t0, 1)

    total = sum(len(v) for v in literature_data.values())
    log(f"  检索完成 ({fetch_time}s), 共 {total} 篇文献")

    if total < 8:
        log(f"  [WARN] 文献不足，跳过训练")
        return {"error": "文献不足", "total_literature": total}

    # 训练
    log(f"  训练 TF-IDF + Logistic Regression...")
    t0 = time.time()
    result = train_classifier(literature_data)
    train_time = round(time.time() - t0, 1)

    if result is None:
        return {"error": "训练失败"}

    pipeline, label_names, top_keywords = result
    accuracy = test_classifier(pipeline, label_names)

    # 拓扑信息
    model_path = str(OUTPUT_DIR / "department_classifier.pkl")
    model_size_kb = round(Path(model_path).stat().st_size / 1024, 1) if Path(model_path).exists() else 0

    metrics = {
        "total_literature": total,
        "fetch_time_sec": fetch_time,
        "train_time_sec": train_time,
        "test_accuracy": round(accuracy, 4),
        "departments_trained": len(label_names),
        "model_path": model_path,
        "model_size_kb": model_size_kb,
        "top_keywords_per_dept": {
            dept: [{"word": w, "weight": round(s, 4)} for w, s in kws[:5]]
            for dept, kws in top_keywords.items()
        },
    }

    log(f"\n  训练完成 ({train_time}s)")
    log(f"  测试准确率: {accuracy:.1%}")
    for dept, dept_cn in DEPARTMENT_NAMES.items():
        kws = top_keywords.get(dept, [])
        log(f"  {dept_cn}: {', '.join(w for w, _ in kws[:5])}")

    save_report({"phase": "1B_department", "metrics": metrics}, "department_train")
    return metrics


# ══════════════════════════════════════════════════════════════
# Phase 2: 全面推理验证
# ══════════════════════════════════════════════════════════════
def run_full_inference(args) -> dict:
    section("Phase 2 — 全面推理验证")

    from data.loader import DataLoader
    from data.preprocessor import HealthcareTokenPreprocessor
    from models.classifier import HealthcareTokenClassifier
    from mcp_server import MedicalDataQAMCPServer

    report = {}
    t_total = time.time()

    # ── 2A: PyTorch 模型推理 ──
    section("  2A — PyTorch 多任务分类器推理")
    loader = DataLoader.instance()
    df_sample = loader.load_sample()

    # 加载模型
    model_path = str(OUTPUT_DIR / "healthcare_token_classifier.pt")
    if not os.path.exists(model_path):
        log("  [WARN] PyTorch 模型文件不存在，跳过")
    else:
        categories = sorted(df_sample["category"].unique().tolist())
        data_types = sorted(df_sample["data_type"].unique().tolist())
        preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)

        model = HealthcareTokenClassifier.load(model_path)
        df = df_sample.reset_index(drop=True)

        # 批量特征（需要先 fit 再 transform，确保 StandardScaler 已拟合）
        t0 = time.time()
        df_clean = preprocessor.clean(df)
        X = preprocessor.fit_transform(df_clean)[0]  # fit_transform 返回 (X, y)，取 X
        preds = model.predict(X)
        infer_time = round(time.time() - t0, 2)
        throughput = int(len(df) / max(infer_time, 0.001))

        # 准确率
        true_level = (df_clean["token_level"].values == "A").astype(np.int64)
        level_acc = float((preds["level_pred"] == true_level).mean())

        true_quality = df_clean["data_quality_score"].values.astype(np.float32)
        quality_mae = float(np.abs(preds["quality_pred"] - true_quality).mean())

        cat_true = np.array([categories.index(c) if c in categories else 0 for c in df_clean["category"]])
        cat_acc = float((preds["category_pred"] == cat_true).mean())

        log(f"  推理规模: {len(df):,} 条, 耗时 {infer_time}s ({throughput:,} 条/秒)")
        log(f"  等级准确率: {level_acc:.4f}")
        log(f"  质量分 MAE:  {quality_mae:.4f}")
        log(f"  科室准确率: {cat_acc:.4f}")

        report["pytorch_infer"] = {
            "records": len(df),
            "infer_time_sec": infer_time,
            "throughput_per_sec": throughput,
            "level_accuracy": round(level_acc, 4),
            "quality_mae": round(quality_mae, 4),
            "category_accuracy": round(cat_acc, 4),
            "model_path": model_path,
        }

    # ── 2B: TF-IDF 科室分类器推理 ──
    section("  2B — KnowS 科室分类器推理")
    from mcp_server import MedicalDataQAMCPServer
    server = MedicalDataQAMCPServer()

    test_texts = [
        ("CT scan shows acute intracerebral hemorrhage in the basal ganglia", "radiology"),
        ("Complete blood count shows leukocytosis with left shift", "laboratory"),
        ("Histopathology reveals invasive ductal carcinoma grade 2", "pathology"),
        ("ECG demonstrates ST-segment elevation in leads V1-V4", "cardiovascular"),
        ("MRI brain with contrast shows multiple enhancing lesions suggestive of metastases", "radiology"),
        ("Patient presents with acute onset right-sided weakness and aphasia, NIHSS 18", "neurology"),
        ("X-ray shows displaced comminuted fracture of the distal radius", "orthopedics"),
        ("Emergency department arrival with GCS 6, intubated, massive transfusion protocol activated", "emergency"),
        ("Neonatal intensive care: 28-week preterm infant with respiratory distress syndrome", "pediatrics"),
        ("Cardiac catheterization reveals 90% LAD stenosis requiring DES deployment", "cardiovascular"),
    ]

    correct = 0
    dept_accuracy = {}
    dept_total = {}
    for text, true_dept in test_texts:
        result = server.classify_department({"text": text, "data_type": "text"})
        pred_dept = result["primary_department"]
        status = "OK" if pred_dept == true_dept else "FAIL"
        log(f"  {status} [{true_dept}] → {pred_dept} (置信度:{result['confidence']:.2f})")
        if pred_dept == true_dept:
            correct += 1
        dept_total[true_dept] = dept_total.get(true_dept, 0) + 1
        if pred_dept == true_dept:
            dept_accuracy[true_dept] = dept_accuracy.get(true_dept, 0) + 1

    dept_acc = {d: round(dept_accuracy.get(d, 0) / dept_total[d], 4) for d in dept_total}
    overall_acc = round(correct / len(test_texts), 4)
    log(f"  总体准确率: {correct}/{len(test_texts)} = {overall_acc:.1%}")

    report["department_infer"] = {
        "test_cases": len(test_texts),
        "correct": correct,
        "overall_accuracy": overall_acc,
        "per_department_accuracy": dept_acc,
    }

    # ── 2C: MCP Server 8工具全面测试 ──
    section("  2C — MCP Server 8工具端到端测试")

    test_records = [
        {"completeness": 95, "accuracy": 92, "timeliness": 88, "compliance": 96, "data_type": "image", "department": "radiology"},
        {"completeness": 78, "accuracy": 85, "timeliness": 70, "compliance": 82, "data_type": "lab"},
        {"completeness": 60, "accuracy": 65, "timeliness": 55, "compliance": 70, "data_type": "text", "department": "pediatrics"},
        {"completeness": 82, "accuracy": 88, "timeliness": 90, "compliance": 85, "data_type": "ecg", "department": "cardiology"},
        {"completeness": 40, "accuracy": 50, "timeliness": 30, "compliance": 45, "data_type": "text"},
    ]

    tool_results = {}

    # Tool 1
    r1 = server.assess_data_quality(test_records)
    tool_results["assess_data_quality"] = {
        "avg_score": r1["average_quality_score"],
        "level_dist": r1["level_distribution"],
        "status": "OK" if r1["average_quality_score"] > 0 else "FAIL",
    }
    log(f"  Tool 1 质量评估: 平均分={r1['average_quality_score']}, 等级分布={r1['level_distribution']}")

    # Tool 2
    r2 = server.classify_department({"text": "CT scan of the chest shows pulmonary nodules requiring biopsy", "data_type": "image"})
    tool_results["classify_department"] = {
        "primary": r2["department_cn"],
        "confidence": r2["confidence"],
        "method": r2["classification_method"],
        "status": "OK",
    }
    log(f"  Tool 2 科室分类: {r2['department_cn']} (置信度={r2['confidence']:.2f}, {r2['classification_method']})")

    # Tool 3
    for score in [93, 82, 67, 45]:
        r3 = server.grade_data_level(score)
        tool_results[f"grade_{score}"] = {"level": r3["level"], "recommend": r3["recommended_use"], "status": "OK"}
    log(f"  Tool 3 等级评定: 93→{server._grade_level(93)}, 82→{server._grade_level(82)}, 67→{server._grade_level(67)}, 45→{server._grade_level(45)}")

    # Tool 4
    r4 = server.generate_quality_report(test_records, "训练验证数据集")
    tool_results["generate_quality_report"] = {
        "avg_quality": r4["summary"]["average_quality"],
        "overall_level": r4["summary"]["overall_level"],
        "weak_dims": [d for d, v in r4["dimension_analysis"].items() if v["weak"]],
        "status": "OK",
    }
    log(f"  Tool 4 质量报告: 等级={r4['summary']['overall_level']}, 薄弱={[d for d, v in r4['dimension_analysis'].items() if v['weak']]}")

    # Tool 5
    r5 = server.search_similar_data({"completeness": 82, "accuracy": 78, "timeliness": 70, "compliance": 85}, top_k=3)
    tool_results["search_similar"] = {"found": r5["total_found"], "top_similarity": r5["top_results"][0]["similarity"] if r5["top_results"] else 0, "status": "OK"}
    log(f"  Tool 5 相似检索: {r5['total_found']} 条, 最高相似度={r5['top_results'][0]['similarity'] if r5['top_results'] else 'N/A'}")

    # Tool 6
    r6 = server.search_medical_evidence("deep learning medical imaging diagnosis", "paper_en", max_results=5)
    if "error" in r6:
        tool_results["search_medical_evidence"] = {"status": "FAIL", "error": r6["error"]}
        log(f"  Tool 6 KnowS检索: FAIL — {r6['error']}")
    else:
        top_if = max((ev.get("impact_factor", 0) or 0) for ev in r6.get("evidences", [{}]))
        tool_results["search_medical_evidence"] = {"found": r6["returned"], "top_if": top_if, "status": "OK"}
        log(f"  Tool 6 KnowS检索: {r6['returned']} 篇, 最高IF={top_if}")

    # Tool 7
    r7 = server.assess_with_evidence(test_records[:2], source="paper_en", evidence_count=3)
    tool_results["assess_with_evidence"] = {
        "avg_quality": r7["quality_assessment"]["average_quality_score"],
        "has_evidence": "evidences" in r7.get("evidence_search", {}) and len(r7["evidence_search"].get("evidences", [])) > 0,
        "status": "OK" if "error" not in r7 else "PARTIAL",
    }
    log(f"  Tool 7 评估+文献联动: 质量分={r7['quality_assessment']['average_quality_score']}")

    # Tool 8
    r8 = server.generate_evidence_based_report(test_records, "训练推理验证", evidence_per_dimension=1)
    if "error" in r8:
        tool_results["evidence_based_report"] = {"status": "FAIL", "error": r8["error"]}
        log(f"  Tool 8 循证报告: FAIL — {r8['error']}")
    else:
        tool_results["evidence_based_report"] = {
            "overall_level": r8["summary"]["overall_level"],
            "total_refs": r8["total_references"],
            "weak_dims": r8["summary"].get("weak_dimensions", []),
            "status": "OK",
        }
        log(f"  Tool 8 循证报告: 等级={r8['summary']['overall_level']}, 引用文献={r8['total_references']} 篇, 薄弱={r8['summary'].get('weak_dimensions', [])}")

    report["mcp_tools"] = tool_results

    # ── 汇总 ──
    total_time = round(time.time() - t_total, 1)
    report["total_time_sec"] = total_time

    tool_status = [v.get("status", "FAIL") for v in tool_results.values()]
    ok_count = sum(1 for s in tool_status if s == "OK")
    log(f"\n  推理总耗时: {total_time}s")
    log(f"  MCP 工具状态: {ok_count}/{len(tool_status)} OK")

    save_report({"phase": "2_inference", "report": report}, "full_inference")
    return report


# ══════════════════════════════════════════════════════════════
# 综合报告
# ══════════════════════════════════════════════════════════════
def generate_summary(phase1a: dict, phase1b: dict, phase2: dict):
    section("综合训练+推理报告")

    log("")
    log("  ╔══════════════════════════════════════════════════╗")
    log("  ║     医疗 AI 模型系统 — 训练+推理完成报告        ║")
    log("  ╚══════════════════════════════════════════════════╝")
    log("")

    # PyTorch 模型
    if "error" not in phase1a:
        log(f"  [PyTorch 多任务分类器]")
        log(f"    等级准确率:  {phase1a.get('level_acc', 'N/A'):.4f}" if isinstance(phase1a.get('level_acc'), float) else f"    等级准确率:  {phase1a.get('level_acc', 'N/A')}")
        log(f"    质量分 MAE:  {phase1a.get('quality_mae', 'N/A'):.4f}" if isinstance(phase1a.get('quality_mae'), float) else f"    质量分 MAE:  {phase1a.get('quality_mae', 'N/A')}")
        log(f"    科室准确率:  {phase1a.get('category_acc', 'N/A'):.4f}" if isinstance(phase1a.get('category_acc'), float) else f"    科室准确率:  {phase1a.get('category_acc', 'N/A')}")
        log(f"    训练耗时:    {phase1a.get('train_time_sec', 'N/A')}s")
        log("")

    # 科室分类器
    if "error" not in phase1b:
        log(f"  [KnowS 科室分类器]")
        log(f"    文献总量:    {phase1b.get('total_literature', 'N/A')} 篇")
        log(f"    测试准确率:  {phase1b.get('test_accuracy', 0):.1%}" if isinstance(phase1b.get('test_accuracy'), float) else f"    测试准确率:  {phase1b.get('test_accuracy', 'N/A')}")
        log(f"    训练耗时:    {phase1b.get('train_time_sec', 'N/A')}s")
        log("")

    # 推理
    pytorch_infer = phase2.get("pytorch_infer", {})
    if pytorch_infer:
        log(f"  [推理 — PyTorch 模型]")
        log(f"    推理规模:    {pytorch_infer.get('records', 0):,} 条")
        log(f"    推理吞吐:    {pytorch_infer.get('throughput_per_sec', 0):,} 条/秒")
        log(f"    等级准确率:  {pytorch_infer.get('level_accuracy', 0):.4f}")
        log(f"    质量分 MAE:  {pytorch_infer.get('quality_mae', 0):.4f}")
        log(f"    科室准确率:  {pytorch_infer.get('category_accuracy', 0):.4f}")
        log("")

    dept_infer = phase2.get("department_infer", {})
    if dept_infer:
        log(f"  [推理 — 科室分类器]")
        log(f"    测试用例:    {dept_infer.get('test_cases', 0)}")
        log(f"    总体准确率:  {dept_infer.get('overall_accuracy', 0):.1%}")
        log("")

    mcp_tools = phase2.get("mcp_tools", {})
    if mcp_tools:
        log(f"  [推理 — MCP Server 8工具]")
        ok = sum(1 for v in mcp_tools.values() if v.get("status") == "OK")
        log(f"    正常: {ok}/{len(mcp_tools)}")
        for name, result in mcp_tools.items():
            status = result.get("status", "?")
            marker = "✓" if status == "OK" else "✗"
            log(f"    {marker} {name}: {status}")

    log("")
    log("  " + "=" * 50)

    # 保存综合报告
    full_report = {
        "timestamp": TIMESTAMP,
        "phase1a_pytorch": phase1a,
        "phase1b_department": phase1b,
        "phase2_inference": phase2,
    }
    path = save_report(full_report, "training_summary")
    log(f"  完整报告: {path}")
    return full_report


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="训练+推理一体化流程")
    # 流程控制
    p.add_argument("--skip-pytorch", action="store_true", help="跳过 PyTorch 模型训练")
    p.add_argument("--skip-knows", action="store_true", help="跳过 KnowS 科室分类器训练")
    p.add_argument("--infer-only", action="store_true", help="仅推理，跳过所有训练")
    p.add_argument("--force-rebuild", action="store_true", help="强制重建数据采样")
    # PyTorch 超参数
    p.add_argument("--pytorch-sample", type=int, default=50000, help="PyTorch 训练抽样数")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early-stop", type=int, default=5)
    # KnowS 参数
    p.add_argument("--knows-per-query", type=int, default=20, help="KnowS 每查询文献数")
    return p.parse_args()


def main():
    args = parse_args()

    log(f"训练+推理一体化开始 [{TIMESTAMP}]")
    log(f"  模式: {'仅推理' if args.infer_only else '完整训练+推理'}")
    phase1a = {}
    phase1b = {}

    if not args.infer_only:
        if not args.skip_pytorch:
            try:
                phase1a = train_pytorch_classifier(args)
            except Exception as e:
                log(f"  [ERROR] PyTorch 训练失败: {e}")
                import traceback
                traceback.print_exc()
                phase1a = {"error": str(e)}
        else:
            log("  [SKIP] PyTorch 模型训练")

        if not args.skip_knows:
            try:
                phase1b = train_department_classifier(args)
            except Exception as e:
                log(f"  [ERROR] 科室分类器训练失败: {e}")
                import traceback
                traceback.print_exc()
                phase1b = {"error": str(e)}
        else:
            log("  [SKIP] KnowS 科室分类器训练")

    # Phase 2: 推理
    try:
        phase2 = run_full_inference(args)
    except Exception as e:
        log(f"  [ERROR] 推理验证失败: {e}")
        import traceback
        traceback.print_exc()
        phase2 = {"error": str(e)}

    # 综合报告
    generate_summary(phase1a, phase1b, phase2)

    log("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
