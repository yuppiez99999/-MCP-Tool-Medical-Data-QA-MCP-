# -*- coding: utf-8 -*-
"""
scripts / train_classifier.py
训练医疗 Token 多任务分类器（等级分类 + 质量回归 + 科室分类）

使用方法：
  cd 18-医疗AI模型系统
  python scripts/train_classifier.py \
      --sample 50000 \
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

from data.loader import DataLoader
from data.preprocessor import HealthcareTokenPreprocessor
from models.classifier import HealthcareTokenClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train Healthcare Token Multi-task Classifier")
    parser.add_argument("--sample", type=int, default=50000, help="训练抽样行数（0 表示使用全部采样数据）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"))
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建采样数据")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] 加载数据集...")
    loader = DataLoader.instance()
    df_sample = loader.load_sample(force_rebuild=args.force_rebuild)
    print(f"  采样数据集: {len(df_sample):,} 条 Token")

    if df_sample.empty:
        print("[ERROR] 数据集为空，请检查数据配置")
        return

    categories = sorted(df_sample["category"].unique().tolist())
    data_types = sorted(df_sample["data_type"].unique().tolist())
    print(f"  科室类别 ({len(categories)}): {categories}")
    print(f"  数据类型 ({len(data_types)}): {data_types}")

    if args.sample > 0 and args.sample < len(df_sample):
        df = df_sample.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"  抽样训练: {len(df):,} 条")
    else:
        df = df_sample.reset_index(drop=True)
        print(f"  使用全部采样数据: {len(df):,} 条")

    print("[2/4] 特征工程...")
    preprocessor = HealthcareTokenPreprocessor(categories=categories, data_types=data_types)
    df_clean = preprocessor.clean(df)
    X_all, y_all = preprocessor.fit_transform(df_clean)
    print(f"  特征维度: {X_all.shape[1]}")
    print(f"  样本数量: {X_all.shape[0]:,}")
    print(f"  等级分布: A级={int(y_all['level'].sum())}, B级={len(y_all['level']) - int(y_all['level'].sum())}")

    indices = np.arange(len(X_all))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42, stratify=y_all["level"])
    X_train, X_val = X_all[idx_train], X_all[idx_val]
    y_train = {k: v[idx_train] for k, v in y_all.items()}
    y_val = {k: v[idx_val] for k, v in y_all.items()}
    print(f"  训练集: {len(X_train):,} 条, 验证集: {len(X_val):,} 条")

    print(f"[3/4] 训练模型 (input_dim={X_train.shape[1]}, num_categories={len(categories)})")
    model = HealthcareTokenClassifier(
        input_dim=X_train.shape[1],
        num_categories=len(categories),
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
    )
    print(f"  设备: {model.device}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  隐藏层数: {args.hidden_layers}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  最大轮数: {args.epochs}")
    print(f"  早停耐心: {args.early_stop}")
    print()

    history = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=args.early_stop,
        verbose=True,
    )

    model_path = os.path.join(args.output_dir, "healthcare_token_classifier.pt")
    model.save(model_path)
    print(f"\n[4/4] 模型已保存至 {model_path}")

    final_metrics = history["val_loss"][-1] if history["val_loss"] else 0.0
    final_level_acc = history["level_acc"][-1] if history["level_acc"] else 0.0
    final_quality_mae = history["quality_mae"][-1] if history["quality_mae"] else 0.0
    final_category_acc = history["category_acc"][-1] if history["category_acc"] else 0.0

    print(f"\n=== 训练结果 ===")
    print(f"  验证损失: {final_metrics:.4f}")
    print(f"  等级分类准确率: {final_level_acc:.4f}")
    print(f"  质量分回归 MAE: {final_quality_mae:.4f}")
    print(f"  科室分类准确率: {final_category_acc:.4f}")
    print("完成。")


if __name__ == "__main__":
    main()
