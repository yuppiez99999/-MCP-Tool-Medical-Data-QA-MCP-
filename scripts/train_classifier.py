# -*- coding: utf-8 -*-
"""
scripts / train_classifier.py
训练医疗 Token 多任务分类器（等级分类 + 质量回归 + 科室分类）

使用方法：
  cd 18-医疗AI模型系统
  python scripts/train_classifier.py \
      --sample 100000 \
      --epochs 50 \
      --batch-size 2048
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from data.loader import HealthcareTokenLoader
from data.preprocessor import HealthcareTokenPreprocessor
from models.classifier import HealthcareTokenClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train Healthcare Token Multi-task Classifier")
    parser.add_argument("--sample", type=int, default=50000, help="训练抽样行数（0 表示全量）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] 加载数据集...")
    loader = HealthcareTokenLoader()
    categories = list(loader.categories.keys())
    data_types = list(loader.data_types.keys())

    df = loader.load_all(sample=args.sample) if args.sample > 0 else loader.load_all()
    print(f"  已加载 {len(df)} 条 Token")

    if df.empty:
        print("[ERROR] 数据集为空，请检查 config.yaml dataset_path")
        return

    print("[2/4] 特征工程...")
    preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
    df_clean = preprocessor.clean(df)
    X_all, y_all = preprocessor.fit_transform(df_clean)

    # 训练 / 验证切分
    indices = np.arange(len(X_all))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42)
    X_train, X_val = X_all[idx_train], X_all[idx_val]
    y_train = {k: v[idx_train] for k, v in y_all.items()}
    y_val = {k: v[idx_val] for k, v in y_all.items()}

    print(f"[3/4] 训练模型 (input_dim={X_train.shape[1]}, num_categories={len(categories)})")
    model = HealthcareTokenClassifier(
        input_dim=X_train.shape[1],
        num_categories=len(categories),
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
    )
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop,
        verbose=True,
    )

    model_path = os.path.join(args.output_dir, "healthcare_token_classifier.pt")
    model.save(model_path)
    print(f"[4/4] 模型已保存至 {model_path}")
    print("完成。")


if __name__ == "__main__":
    main()
