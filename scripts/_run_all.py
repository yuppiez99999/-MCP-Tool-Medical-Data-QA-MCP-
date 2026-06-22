# -*- coding: utf-8 -*-
"""批量登记判定+质量评估+科室分类+资产估值"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.loader import HealthcareTokenLoader
from models.valuation_engine import AssetValuationEngine
from models.classifier import HealthcareTokenClassifier
from data.preprocessor import HealthcareTokenPreprocessor

OUT = "outputs"

def main():
    # ============ 1. 加载数据 ============
    print("[1/4] 加载数据集...")
    loader = HealthcareTokenLoader()
    df = loader.load_all(sample=100000)
    print(f"  已加载 {len(df):,} 条 Token")
    summary = loader.summary()
    print(f"  质量分: {summary['quality_min']:.1f}~{summary['quality_max']:.1f}, 均值{summary['quality_mean']:.1f}")
    print(f"  等级分布: A={summary['level_counts']['A']:,}, B={summary['level_counts']['B']:,}")

    # ============ 2. 科室分类预测 ============
    print("\n[2/4] 科室分类预测...")
    model_path = os.path.join(OUT, "healthcare_token_classifier.pt")
    model = HealthcareTokenClassifier.load(model_path)
    categories = list(loader.categories.keys())
    data_types = list(loader.data_types.keys())
    preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
    df_clean = preprocessor.clean(df)
    X, _ = preprocessor.fit_transform(df_clean)
    preds = model.predict(X)
    level_pred = preds['level_pred']
    quality_pred = preds['quality_pred']
    category_pred = preds['category_pred']
    df['pred_level'] = ['A' if l == 1 else 'B' for l in level_pred]
    df['pred_quality'] = quality_pred
    df['pred_category'] = [categories[c] for c in category_pred]
    print(f"  等级判定: A={sum(level_pred)}, B={sum(1-level_pred)}")
    print(f"  质量评估: 预测均值 {quality_pred.mean():.1f}, 实际均值 {df['data_quality_score'].mean():.1f}")
    level_acc = (level_pred == 1).mean()
    quality_mae = abs(quality_pred - df['data_quality_score']).mean()
    print(f"  质量MAE: {quality_mae:.2f}")

    # ============ 3. 资产估值 ============
    print("\n[3/4] 资产估值...")
    engine = AssetValuationEngine(category_counts=summary.get("category_counts", {}))
    valued = engine.value_dataframe(df)
    vs = engine.summary(valued)

    print(f"  合格Token数: {vs['total_tokens']:,}")
    print(f"  总价值(单条): {vs['total_value']:,.2f}")
    print(f"  企业批量价:   {vs['total_enterprise_value']:,.2f}")
    print(f"  每条约:       {vs['avg_value_per_token']:.2f}")
    print(f"  按级价值:")
    for lv in vs['by_level']:
        print(f"    等级{lv['token_level']}: {lv['count']:,}条, 总值{lv['total_value']:,.0f}, 均价{lv['avg_value']:.2f}")

    # ============ 4. 保存 ============
    print("\n[4/4] 保存结果...")
    detail_path = os.path.join(OUT, "valuation_detail_full.csv")
    summary_path = os.path.join(OUT, "valuation_summary_full.csv")
    valued.to_csv(detail_path, index=False, encoding="utf-8")
    by_cat = pd.DataFrame(vs.get("by_category", []))
    by_cat.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"  明细: {detail_path}")
    print(f"  汇总: {summary_path}")
    print("\n完成。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
