"""
科室分类模型评测脚本 — 50+条测试用例
用于评估文献驱动的科室分类模型效果
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import MedicalDataQAMCPServer

DEPT_NAMES = {
    "radiology": "放射科",
    "pathology": "病理科",
    "neurology": "神经内科",
    "cardiovascular": "心血管科",
    "laboratory": "检验科",
    "orthopedics": "骨科",
    "pediatrics": "儿科",
    "emergency": "急诊科",
}

# 50+ 条测试用例 (文本, 正确科室)
TEST_CASES = [
    # ===== 放射科 (8条) =====
    ("CT scan of the brain shows acute ischemic stroke in the left hemisphere", "radiology"),
    ("MRI of the abdomen with contrast showing liver mass suspicious for hepatocellular carcinoma", "radiology"),
    ("Chest X-ray reveals bilateral pulmonary infiltrates consistent with pneumonia", "radiology"),
    ("CT angiography of coronary arteries showing significant stenosis in LAD", "radiology"),
    ("Mammogram demonstrating suspicious microcalcifications in the right breast", "radiology"),
    ("Ultrasound of thyroid showing hypoechoic nodule with irregular borders", "radiology"),
    ("PET-CT scan demonstrating hypermetabolic lymph nodes in the mediastinum", "radiology"),
    ("Bone scan showing multiple areas of increased uptake suggestive of metastases", "radiology"),

    # ===== 病理科 (8条) =====
    ("Histopathology of colon biopsy showing well-differentiated adenocarcinoma", "pathology"),
    ("Fine needle aspiration cytology of thyroid nodule revealing papillary carcinoma", "pathology"),
    ("Immunohistochemistry staining positive for ER and PR in breast cancer specimen", "pathology"),
    ("Bone marrow biopsy showing hypercellularity with increased blasts consistent with AML", "pathology"),
    ("Histological examination of skin biopsy revealing basal cell carcinoma", "pathology"),
    ("Frozen section analysis of sentinel lymph node during melanoma surgery", "pathology"),
    ("Cytopathology of pleural effusion showing malignant cells consistent with adenocarcinoma", "pathology"),
    ("Molecular pathology testing for EGFR mutations in non-small cell lung cancer", "pathology"),

    # ===== 神经内科 (8条) =====
    ("Electroencephalogram showing generalized spike and wave discharges consistent with epilepsy", "neurology"),
    ("Patient presenting with progressive memory decline and cognitive impairment suspected Alzheimer disease", "neurology"),
    ("Parkinson disease patient with resting tremor bradykinesia and rigidity", "neurology"),
    ("Multiple sclerosis patient with demyelinating lesions on MRI and oligoclonal bands in CSF", "neurology"),
    ("Acute ischemic stroke patient with right-sided weakness and aphasia", "neurology"),
    ("Migraine headache with aura visual disturbances and unilateral throbbing pain", "neurology"),
    ("Guillain-Barre syndrome with ascending paralysis and areflexia", "neurology"),
    ("Amyotrophic lateral sclerosis with progressive muscle weakness and fasciculations", "neurology"),

    # ===== 心血管科 (8条) =====
    ("Electrocardiogram demonstrating ST-elevation myocardial infarction in anterior leads", "cardiovascular"),
    ("Echocardiogram showing reduced ejection fraction of 35% consistent with heart failure", "cardiovascular"),
    ("Coronary angiography revealing triple vessel disease with significant stenosis", "cardiovascular"),
    ("Patient with essential hypertension on multiple antihypertensive medications", "cardiovascular"),
    ("Atrial fibrillation with rapid ventricular response on ECG requiring rate control", "cardiovascular"),
    ("Atherosclerotic cardiovascular disease with carotid artery stenosis on Doppler ultrasound", "cardiovascular"),
    ("Cardiac catheterization for evaluation of chest pain and coronary artery disease", "cardiovascular"),
    ("Congestive heart failure exacerbation with pulmonary edema and elevated BNP", "cardiovascular"),

    # ===== 检验科 (8条) =====
    ("Complete blood count showing leukocytosis with neutrophilia and left shift", "laboratory"),
    ("Blood chemistry panel demonstrating elevated liver enzymes ALT and AST", "laboratory"),
    ("Urinalysis showing proteinuria hematuria and casts consistent with nephrotic syndrome", "laboratory"),
    ("Clinical chemistry test results showing hyperglycemia and elevated HbA1c", "laboratory"),
    ("Coagulation panel with prolonged PT and INR consistent with warfarin therapy", "laboratory"),
    ("Blood gas analysis showing metabolic acidosis with compensatory respiratory alkalosis", "laboratory"),
    ("Immunoassay for cardiac troponin I showing elevated levels indicating myocardial injury", "laboratory"),
    ("Microbiology culture of wound growing methicillin-resistant Staphylococcus aureus", "laboratory"),

    # ===== 骨科 (6条) =====
    ("X-ray showing comminuted fracture of the femoral shaft following trauma", "orthopedics"),
    ("MRI of lumbar spine demonstrating L4-L5 disc herniation with nerve root compression", "orthopedics"),
    ("Total knee arthroplasty for end-stage osteoarthritis of the knee joint", "orthopedics"),
    ("Spinal fusion surgery for degenerative disc disease with instability", "orthopedics"),
    ("Orthopedic implant for fracture fixation using intramedullary nail", "orthopedics"),
    ("Rotator cuff tear of the shoulder confirmed on MRI requiring arthroscopic repair", "orthopedics"),

    # ===== 儿科 (6条) =====
    ("Neonatal intensive care unit patient born prematurely at 28 weeks gestation", "pediatrics"),
    ("Pediatric patient with acute lymphoblastic leukemia undergoing induction chemotherapy", "pediatrics"),
    ("Child with measles infection presenting with fever cough and maculopapular rash", "pediatrics"),
    ("Infant with failure to thrive and feeding difficulties requiring nutritional support", "pediatrics"),
    ("Pediatric asthma exacerbation with wheezing and shortness of breath", "pediatrics"),
    ("Newborn screening test positive for phenylketonuria requiring dietary management", "pediatrics"),

    # ===== 急诊科 (6条) =====
    ("Emergency department patient with severe trauma and hypovolemic shock from motor vehicle accident", "emergency"),
    ("Cardiac arrest patient receiving CPR and advanced cardiovascular life support in ER", "emergency"),
    ("Sepsis patient with hypotension tachycardia and elevated lactate requiring ICU admission", "emergency"),
    ("Acute respiratory distress syndrome patient requiring mechanical ventilation in emergency", "emergency"),
    ("Emergency resuscitation of critically ill patient with multiple organ dysfunction syndrome", "emergency"),
    ("Toxicology patient with drug overdose and altered mental status in emergency room", "emergency"),
]


def run_evaluation():
    """运行完整评测"""
    print("=" * 70)
    print("  医疗数据科室分类模型评测 — 50+条测试用例")
    print("=" * 70)

    server = MedicalDataQAMCPServer()
    model_loaded = server.department_model is not None
    print(f"\n模型状态: {'已加载' if model_loaded else '未加载(使用规则引擎)'}")
    if model_loaded:
        print(f"  类型: {server.department_model['model_type']}")
        print(f"  训练样本: {server.department_model['training_samples']}")

    results = []
    dept_correct = {}
    dept_total = {}

    print(f"\n{'序号':<4} {'科室':<8} {'预测':<8} {'置信度':<8} {'方法':<8} {'结果':<6}")
    print("-" * 70)

    for i, (text, true_dept) in enumerate(TEST_CASES, 1):
        record = {"text": text, "data_type": "text"}
        result = server.classify_department(record)
        pred_dept = result["primary_department"]
        conf = result["confidence"]
        method = "ML" if result["classification_method"] == "ml_model" else "规则"
        correct = (pred_dept == true_dept)
        status = "OK" if correct else "FAIL"

        results.append({
            "index": i,
            "text": text,
            "true_dept": true_dept,
            "true_dept_cn": DEPT_NAMES.get(true_dept, true_dept),
            "pred_dept": pred_dept,
            "pred_dept_cn": result["department_cn"],
            "confidence": conf,
            "method": method,
            "correct": correct,
        })

        dept_total[true_dept] = dept_total.get(true_dept, 0) + 1
        if correct:
            dept_correct[true_dept] = dept_correct.get(true_dept, 0) + 1

        true_cn = DEPT_NAMES.get(true_dept, true_dept)
        pred_cn = result["department_cn"]
        print(f"{i:<4} {true_cn:<6} {pred_cn:<6} {conf:.2%}   {method:<6} {status:<6}")

    # 总体统计
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total

    print("\n" + "=" * 70)
    print(f"  总体结果: {correct_count}/{total} = {accuracy:.1%}")
    print("=" * 70)

    # 各科室准确率
    print("\n  各科室准确率:")
    print(f"  {'科室':<10} {'正确':<6} {'总数':<6} {'准确率':<8}")
    print("  " + "-" * 35)
    for dept in sorted(dept_total.keys()):
        cn = DEPT_NAMES.get(dept, dept)
        c = dept_correct.get(dept, 0)
        t = dept_total[dept]
        acc = c / t if t > 0 else 0
        bar = "█" * int(acc * 20)
        print(f"  {cn:<8} {c:<6} {t:<6} {acc:.1%}  {bar}")

    # 按方法统计
    ml_results = [r for r in results if r["method"] == "ML"]
    rule_results = [r for r in results if r["method"] == "规则"]
    if ml_results:
        ml_correct = sum(1 for r in ml_results if r["correct"])
        print(f"\n  ML模型分类: {ml_correct}/{len(ml_results)} = {ml_correct/len(ml_results):.1%}")
    if rule_results:
        rule_correct = sum(1 for r in rule_results if r["correct"])
        print(f"  规则引擎分类: {rule_correct}/{len(rule_results)} = {rule_correct/len(rule_results):.1%}")

    # 置信度分析
    confidences = [r["confidence"] for r in results]
    print(f"\n  置信度统计:")
    print(f"    平均置信度: {sum(confidences)/len(confidences):.2%}")
    print(f"    最高置信度: {max(confidences):.2%}")
    print(f"    最低置信度: {min(confidences):.2%}")

    # 错误案例分析
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  错误案例 ({len(wrong)}条):")
        for w in wrong:
            print(f"    [{w['index']}] 真实:{w['true_dept_cn']} → 预测:{w['pred_dept_cn']} ({w['confidence']:.1%})")
            print(f"        {w['text'][:70]}...")

    print("\n" + "=" * 70)

    # 保存结果
    output_path = Path(__file__).parent.parent / "outputs" / "evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "correct": correct_count,
            "accuracy": accuracy,
            "per_department": {
                dept: {
                    "name_cn": DEPT_NAMES.get(dept, dept),
                    "correct": dept_correct.get(dept, 0),
                    "total": dept_total[dept],
                    "accuracy": dept_correct.get(dept, 0) / dept_total[dept] if dept_total[dept] > 0 else 0,
                }
                for dept in dept_total
            },
            "model_loaded": model_loaded,
            "test_cases": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"  详细结果已保存至: {output_path}")
    print("=" * 70)

    return accuracy


if __name__ == "__main__":
    run_evaluation()
