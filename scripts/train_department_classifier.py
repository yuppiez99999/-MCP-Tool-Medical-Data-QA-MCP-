"""文献驱动的科室分类模型训练脚本

通过KnowS API检索8个科室的代表性医学文献，
提取文本特征，训练TF-IDF + Logistic Regression分类器。
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import MedicalDataQAMCPServer


DEPARTMENT_QUERIES = {
    "radiology": [
        "radiology imaging CT MRI diagnosis",
        "X-ray computed tomography radiological",
        "medical imaging tumor detection",
    ],
    "pathology": [
        "pathology histopathology biopsy cancer",
        "tumor pathology diagnosis biomarker",
        "histological examination tissue",
    ],
    "neurology": [
        "neurology stroke Alzheimer Parkinson",
        "neuroscience brain nervous system",
        "neurodegenerative disease EEG",
    ],
    "cardiovascular": [
        "cardiology cardiovascular heart disease",
        "myocardial infarction ECG hypertension",
        "atherosclerosis coronary artery",
    ],
    "laboratory": [
        "clinical laboratory diagnosis biomarker",
        "blood test clinical chemistry",
        "lab medicine diagnostic testing",
    ],
    "orthopedics": [
        "orthopedics fracture bone joint",
        "spine surgery musculoskeletal",
        "orthopaedic trauma implant",
    ],
    "pediatrics": [
        "pediatrics neonatal child health",
        "paediatric infant disease",
        "children hospital pediatric care",
    ],
    "emergency": [
        "emergency medicine trauma critical",
        "critical care ICU emergency",
        "acute care resuscitation shock",
    ],
}

DEPARTMENT_NAMES = {
    "radiology": "放射科",
    "pathology": "病理科",
    "neurology": "神经内科",
    "cardiovascular": "心血管科",
    "laboratory": "检验科",
    "orthopedics": "骨科",
    "pediatrics": "儿科",
    "emergency": "急诊科",
}

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_literature(server, max_per_query: int = 20) -> dict:
    """从KnowS API检索各科室文献"""
    all_data = {}
    total_fetched = 0

    for dept, queries in DEPARTMENT_QUERIES.items():
        dept_papers = []
        seen_ids = set()
        for query in queries:
            result = server.search_medical_evidence(query, "paper_en", max_per_query)
            if "error" in result:
                print(f"  检索失败 [{dept}] [{query}]: {result['error']}")
                continue

            for ev in result.get("evidences", []):
                pid = ev.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    title = ev.get("title", "")
                    journal = ev.get("journal", "")
                    abstract = ev.get("abstract", "")
                    study_type = ev.get("study_type", "")
                    text = f"{title} {journal} {abstract} {study_type}".strip()
                    dept_papers.append({
                        "id": pid,
                        "text": text,
                        "title": title,
                    })
            time.sleep(0.2)

        all_data[dept] = dept_papers
        total_fetched += len(dept_papers)
        print(f"  {DEPARTMENT_NAMES[dept]}: {len(dept_papers)} 篇")

    print(f"\n总计获取 {total_fetched} 篇文献")
    return all_data


def train_classifier(literature_data: dict):
    """训练TF-IDF + Logistic Regression分类器"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
    except ImportError:
        print("错误: 缺少scikit-learn依赖，请运行: pip install scikit-learn")
        return None

    texts = []
    labels = []
    label_names = []

    for dept, papers in literature_data.items():
        for p in papers:
            texts.append(p["text"])
            labels.append(dept)
        label_names.append(dept)

    print(f"\n训练样本: {len(texts)} 条")
    print(f"科室数量: {len(label_names)}")
    for dept in label_names:
        cnt = labels.count(dept)
        print(f"  {DEPARTMENT_NAMES[dept]}: {cnt} 条")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",
        )),
    ])

    if len(texts) >= 24:
        cv = min(5, len(texts) // len(label_names))
        if cv >= 2:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, texts, labels, cv=skf, scoring="accuracy")
            print(f"\n交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")
            print(f"  各折: {[f'{s:.2f}' for s in scores]}")

    pipeline.fit(texts, labels)

    label_names = list(pipeline.classes_)
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coef = pipeline.named_steps["clf"].coef_

    top_keywords = {}
    for i, dept in enumerate(label_names):
        top_indices = np.argsort(coef[i])[-20:][::-1]
        top_keywords[dept] = [
            (feature_names[idx], float(coef[i][idx]))
            for idx in top_indices
        ]

    model_path = OUTPUT_DIR / "department_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "label_names": label_names,
            "top_keywords": top_keywords,
            "training_samples": len(texts),
            "model_type": "tfidf_logistic_regression",
            "training_source": "knows_paper_en",
        }, f)

    print(f"\n模型已保存至: {model_path}")
    print(f"模型大小: {model_path.stat().st_size / 1024:.1f} KB")

    print("\n=== 各科室Top关键词 ===")
    for dept in label_names:
        keywords = top_keywords.get(dept, [])
        words = [w for w, s in keywords[:10]]
        print(f"  {DEPARTMENT_NAMES[dept]}: {', '.join(words)}")

    return pipeline, label_names, top_keywords


def test_classifier(pipeline, label_names):
    """测试分类效果"""
    test_cases = [
        ("CT scan of the brain shows acute ischemic stroke in the left hemisphere", "neurology"),
        ("Pathological examination reveals invasive ductal carcinoma of the breast", "pathology"),
        ("X-ray imaging shows comminuted fracture of the femoral shaft", "orthopedics"),
        ("Emergency department patient with severe trauma and hemorrhagic shock", "emergency"),
        ("ECG demonstrates ST-elevation myocardial infarction", "cardiovascular"),
        ("Clinical laboratory blood test shows elevated troponin levels", "laboratory"),
        ("Pediatric patient with acute respiratory infection and fever", "pediatrics"),
        ("MRI radiological evaluation of brain tumor using contrast enhancement", "radiology"),
        ("Histopathology of colon biopsy showing adenocarcinoma", "pathology"),
        ("Cardiac catheterization for coronary artery disease", "cardiovascular"),
    ]

    correct = 0
    print("\n=== 分类测试 ===")
    for text, true_dept in test_cases:
        pred = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])
        conf = proba.max()
        status = "OK" if pred == true_dept else "FAIL"
        if pred == true_dept:
            correct += 1
        true_name = DEPARTMENT_NAMES.get(true_dept, true_dept)
        pred_name = DEPARTMENT_NAMES.get(pred, pred)
        print(f"  {status} [{true_name}] -> {pred_name} (置信度: {conf:.2%})")

    print(f"\n测试准确率: {correct}/{len(test_cases)} = {correct/len(test_cases):.1%}")
    return correct / len(test_cases)


def main():
    api_key = os.environ.get("KNOWS_API_KEY", "")
    if not api_key:
        print("错误: 请设置 KNOWS_API_KEY 环境变量")
        sys.exit(1)

    print("=" * 60)
    print("文献驱动科室分类模型训练")
    print("=" * 60)

    server = MedicalDataQAMCPServer()

    print("\n[1/3] 从KnowS API检索文献...")
    literature_data = fetch_literature(server, max_per_query=20)

    total = sum(len(v) for v in literature_data.values())
    if total < 8:
        print("错误: 获取的文献太少，无法训练")
        sys.exit(1)

    print("\n[2/3] 训练TF-IDF + Logistic Regression分类器...")
    result = train_classifier(literature_data)
    if result is None:
        sys.exit(1)

    pipeline, label_names, top_keywords = result

    print("\n[3/3] 测试分类效果...")
    accuracy = test_classifier(pipeline, label_names)

    print("\n" + "=" * 60)
    print(f"训练完成！准确率: {accuracy:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
