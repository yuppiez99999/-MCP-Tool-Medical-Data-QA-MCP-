# -*- coding: utf-8 -*-
"""
scripts / run_valuation.py
批量数据资产估值脚本

使用方法：
  cd 18-医疗AI模型系统
  python scripts/run_valuation.py --sample 100000
"""

import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from data.loader import HealthcareTokenLoader
from models.valuation_engine import AssetValuationEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Run Asset Valuation")
    parser.add_argument("--sample", type=int, default=100000, help="抽样行数 (0 = 全量)")
    parser.add_argument("--data-version", type=str, default="100M", help="数据集版本标签（输出文件命名用）")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/3] 加载数据集...")
    loader = HealthcareTokenLoader()
    df = loader.load_all(sample=args.sample) if args.sample > 0 else loader.load_all()
    print(f"  已加载 {len(df)} 条 Token")

    if df.empty:
        print("[ERROR] 数据集为空")
        return

    summary = loader.summary()
    print(f"  质量分范围: {summary['quality_min']:.2f} - {summary['quality_max']:.2f}")

    print("[2/3] 估值引擎启动...")
    engine = AssetValuationEngine(category_counts=summary.get("category_counts", {}))
    valued_df = engine.value_dataframe(df)

    val_summary = engine.summary(valued_df)
    print("[3/3] 资产估值汇总：")
    print(f"  合格 Token 数：{val_summary['total_tokens']:,}")
    print(f"  总价值（单条参考价）：¥ {val_summary['total_value']:,.2f}")
    print(f"  总价值（企业批量价）：¥ {val_summary['total_enterprise_value']:,.2f}")
    print(f"  平均每条价值：¥ {val_summary['avg_value_per_token']:.2f}")

    print(f"  数据集版本：{args.data_version}")

    # 保存
    version_tag = args.data_version.replace("/", "_")
    detail_path = os.path.join(args.output_dir, f"valuation_detail_{version_tag}.csv")
    summary_path = os.path.join(args.output_dir, f"valuation_summary_{version_tag}.csv")
    valued_df.to_csv(detail_path, index=False, encoding="utf-8")
    by_cat = pd.DataFrame(val_summary.get("by_category", []))
    by_cat.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n明细已保存至：{detail_path}")
    print(f"汇总已保存至：{summary_path}")


if __name__ == "__main__":
    main()
