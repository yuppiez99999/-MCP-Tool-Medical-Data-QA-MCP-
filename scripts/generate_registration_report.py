# -*- coding: utf-8 -*-
"""
scripts / generate_registration_report.py
生成北数所数据资产登记报告（Markdown + JSON）

使用方法：
  cd 18-医疗AI模型系统
  python scripts/generate_registration_report.py --sample 100000
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

import yaml

from data.loader import HealthcareTokenLoader
from models.valuation_engine import AssetValuationEngine
from modules.data_exchange import DataExchangeRegistrar


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Data Exchange Registration Report")
    parser.add_argument("--sample", type=int, default=50000, help="抽样行数 (0 = 全量)")
    parser.add_argument("--domain", type=str, default="healthcare", help="数据域标签 (输出文件命名)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"))
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[1/3] 加载数据集... (domain={args.domain})")
    loader = HealthcareTokenLoader(config_path=args.config)
    df = loader.load_all(sample=args.sample) if args.sample > 0 else loader.load_all()
    print(f"  已加载 {len(df)} 条 Token")
    summary = loader.summary()

    print("[2/3] 资产估值...")
    engine = AssetValuationEngine(
        config_path=args.config,
        category_counts=summary.get("category_counts", {}),
    )
    valued = engine.value_dataframe(df)
    val_summary = engine.summary(valued)

    print("[3/3] 生成登记报告...")
    registrar = DataExchangeRegistrar(
        config=config,
        dataset_summary=summary,
        valuation_summary=val_summary,
    )

    md_path = os.path.join(args.output_dir, f"registration_report_{args.domain}.md")
    json_path = os.path.join(args.output_dir, f"registration_report_{args.domain}.json")
    registrar.generate_markdown(raw_df_sample=df, output_path=md_path)
    registrar.generate_json(raw_df_sample=df, output_path=json_path)

    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")
    print("完成。")


if __name__ == "__main__":
    main()
