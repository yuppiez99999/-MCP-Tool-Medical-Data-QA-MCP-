# -*- coding: utf-8 -*-
"""
models / classifier.py
医疗健康 Token 多任务学习分类器

架构：
  输入特征 (N 维)
       │
       ▼
  Linear(64) + BatchNorm + ReLU + Dropout
       │
  Linear(128) + BatchNorm + ReLU + Dropout
       │
  ┌────┼────┐
  ▼    ▼    ▼
Token 质量分  科室
等级   回归   分类
 (2)   (1)   (8)

多任务学习 + 早停 + 模型保存/加载
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# 模型架构
# -----------------------------
class _MultiTaskNet(nn.Module):
    """多任务学习网络（Token 等级二分类 + 质量分回归 + 科室多分类）"""

    def __init__(self, input_dim: int, hidden_dim: int, num_categories: int, dropout: float = 0.2, hidden_layers: int = 3):
        super().__init__()
        # 共享主干
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
            hidden_dim = hidden_dim // 2
        self.backbone = nn.Sequential(*layers)
        last_dim = dim

        # 三任务头
        self.head_level = nn.Linear(last_dim, 2)  # 等级二分类（A/B）
        self.head_quality = nn.Linear(last_dim, 1)  # 质量分回归
        self.head_category = nn.Linear(last_dim, num_categories)  # 科室多分类

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.backbone(x)
        return {
            "level": self.head_level(shared),
            "quality": self.head_quality(shared).squeeze(-1),
            "category": self.head_category(shared),
        }


# -----------------------------
# 对外封装
# -----------------------------
class HealthcareTokenClassifier:
    """医疗 Token 多任务分类器 —— 面向医疗 AI 模型训练场景"""

    def __init__(
        self,
        input_dim: int,
        num_categories: int,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        loss_weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.loss_weights = loss_weights or {"level": 1.0, "quality": 0.5, "category": 1.0}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MultiTaskNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_categories=num_categories,
            dropout=dropout,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self._trained = False
        self._train_history: List[Dict[str, float]] = []

    # ---------------------------
    # 训练
    # ---------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 2048,
        early_stop_patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        self.model.train()
        train_ds = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train["level"].astype(np.int64)),
            torch.from_numpy(y_train["quality"].astype(np.float32)),
            torch.from_numpy(y_train["category"].astype(np.int64)),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        best_val_loss = float("inf")
        patience_counter = 0
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "level_acc": [],
            "quality_mae": [],
            "category_acc": [],
        }

        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_loader:
                xb, lvl, qlt, cat = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                preds = self.model(xb)

                lvl_loss = F.cross_entropy(preds["level"], lvl)
                qlt_loss = F.l1_loss(preds["quality"], qlt)
                cat_loss = F.cross_entropy(preds["category"], cat)

                loss = (
                    self.loss_weights["level"] * lvl_loss
                    + self.loss_weights["quality"] * qlt_loss
                    + self.loss_weights["category"] * cat_loss
                )
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(avg_train_loss)

            # 验证集
            if X_val is not None and y_val is not None:
                val_metrics = self._evaluate(X_val, y_val, batch_size=batch_size)
                for k, v in val_metrics.items():
                    history[k].append(v)
                val_loss = val_metrics["val_loss"]
                if verbose:
                    print(
                        f"[Epoch {epoch + 1:3d}] train_loss={avg_train_loss:.4f} "
                        f"val_loss={val_loss:.4f} "
                        f"level_acc={val_metrics['level_acc']:.3f} "
                        f"quality_mae={val_metrics['quality_mae']:.3f} "
                        f"category_acc={val_metrics['category_acc']:.3f}"
                    )
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(f"[EarlyStop] 验证集 {early_stop_patience} 轮未提升，停止训练")
                        break
            else:
                if verbose:
                    print(f"[Epoch {epoch + 1:3d}] train_loss={avg_train_loss:.4f}")

        self._trained = True
        self._train_history.append({k: v[-1] if v else 0.0 for k, v in history.items()})
        return history

    # ---------------------------
    # 验证
    # ---------------------------
    def _evaluate(
        self,
        X: np.ndarray,
        y: Dict[str, np.ndarray],
        batch_size: int,
    ) -> Dict[str, float]:
        self.model.eval()
        preds_level, preds_quality, preds_category = self._batch_predict(X, batch_size)
        level_true = torch.from_numpy(y["level"].astype(np.int64)).to(self.device)
        quality_true = torch.from_numpy(y["quality"].astype(np.float32)).to(self.device)
        category_true = torch.from_numpy(y["category"].astype(np.int64)).to(self.device)

        lvl_loss = F.cross_entropy(preds_level, level_true).item()
        qlt_loss = F.l1_loss(preds_quality, quality_true).item()
        cat_loss = F.cross_entropy(preds_category, category_true).item()

        level_acc = float((preds_level.argmax(dim=1) == level_true).float().mean().item())
        category_acc = float((preds_category.argmax(dim=1) == category_true).float().mean().item())
        quality_mae = float(F.l1_loss(preds_quality, quality_true).item())

        val_loss = (
            self.loss_weights["level"] * lvl_loss
            + self.loss_weights["quality"] * qlt_loss
            + self.loss_weights["category"] * cat_loss
        )
        return {
            "val_loss": float(val_loss),
            "level_acc": level_acc,
            "quality_mae": quality_mae,
            "category_acc": category_acc,
        }

    def _batch_predict(self, X: np.ndarray, batch_size: int):
        preds_level_list, preds_qlt_list, preds_cat_list = [], [], []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                xb = torch.from_numpy(X[start:end].astype(np.float32)).to(self.device)
                out = self.model(xb)
                preds_level_list.append(out["level"])
                preds_qlt_list.append(out["quality"])
                preds_cat_list.append(out["category"])
        return (
            torch.cat(preds_level_list, dim=0),
            torch.cat(preds_qlt_list, dim=0),
            torch.cat(preds_cat_list, dim=0),
        )

    # ---------------------------
    # 预测接口
    # ---------------------------
    def predict(self, X: np.ndarray, batch_size: int = 2048) -> Dict[str, np.ndarray]:
        self.model.eval()
        pl, pq, pc = self._batch_predict(X, batch_size)
        level_proba = torch.softmax(pl, dim=1).cpu().numpy()
        quality = pq.cpu().numpy()
        category_proba = torch.softmax(pc, dim=1).cpu().numpy()
        return {
            "level_proba": level_proba,
            "level_pred": np.argmax(level_proba, axis=1),  # 1=A, 0=B
            "quality_pred": quality,
            "category_proba": category_proba,
            "category_pred": np.argmax(category_proba, axis=1),
        }

    # ---------------------------
    # 保存/加载
    # ---------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_dim": self.input_dim,
                "num_categories": self.num_categories,
                "loss_weights": self.loss_weights,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "HealthcareTokenClassifier":
        checkpoint = torch.load(path, map_location=device or ("cpu"))
        obj = cls(
            input_dim=checkpoint["input_dim"],
            num_categories=checkpoint["num_categories"],
            loss_weights=checkpoint["loss_weights"],
            device=device,
        )
        obj.model.load_state_dict(checkpoint["model_state_dict"])
        obj.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        obj._trained = True
        return obj

    @property
    def is_trained(self) -> bool:
        return self._trained
