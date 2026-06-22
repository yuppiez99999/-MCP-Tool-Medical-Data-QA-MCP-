# -*- coding: utf-8 -*-
"""
modules / auth_manager.py
用户认证与套餐管理模块

核心功能：
1. 用户注册、登录、认证
2. 套餐管理（基础版/专业版/旗舰版/定制版）
3. API 调用次数限制与验证
4. 权限管理

套餐配置：
- 基础版（¥180,000/年）：500次/天 API 调用，4个科室
- 专业版（¥380,000/年）：2000次/天 API 调用，8个科室
- 旗舰版（¥680,000/年）：10000次/天 API 调用，全部科室+自定义
- 定制版（面议）：按需配置
"""

import os
import sqlite3
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yaml


def _load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AuthManager:
    """用户认证与套餐管理模块"""

    PLANS = {
        "basic": {
            "name": "基础版",
            "price": 180000,
            "daily_limit": 500,
            "max_tokens": 50000,
            "allowed_categories": ["laboratory", "orthopedics", "pediatrics", "emergency"],
            "allowed_data_types": ["lab"],
            "support_hours": "工作日 8 小时",
            "report_frequency": "季度",
            "training_count": 1,
        },
        "professional": {
            "name": "专业版",
            "price": 380000,
            "daily_limit": 2000,
            "max_tokens": 200000,
            "allowed_categories": ["radiology", "pathology", "neurology", "cardiology",
                                  "laboratory", "orthopedics", "pediatrics", "emergency"],
            "allowed_data_types": ["image", "text", "lab", "ecg", "pathology"],
            "support_hours": "7×12 小时",
            "report_frequency": "月度",
            "training_count": 4,
        },
        "enterprise": {
            "name": "旗舰版",
            "price": 680000,
            "daily_limit": 10000,
            "max_tokens": 500000,
            "allowed_categories": ["radiology", "pathology", "neurology", "cardiology",
                                  "laboratory", "orthopedics", "pediatrics", "emergency"],
            "allowed_data_types": ["image", "text", "ecg", "lab", "pathology",
                                   "genetic", "vital", "other"],
            "support_hours": "7×24 小时",
            "report_frequency": "实时",
            "training_count": -1,
        },
        "custom": {
            "name": "定制版",
            "price": 0,
            "daily_limit": -1,
            "max_tokens": -1,
            "allowed_categories": ["radiology", "pathology", "neurology", "cardiology",
                                  "laboratory", "orthopedics", "pediatrics", "emergency"],
            "allowed_data_types": ["image", "text", "ecg", "lab", "pathology",
                                   "genetic", "vital", "other"],
            "support_hours": "专属",
            "report_frequency": "按需",
            "training_count": -1,
        },
    }

    API_RATES = {
        "predict_level": {"basic": 0.05, "professional": 0.04, "enterprise": 0.03, "custom": 0.02},
        "predict_quality": {"basic": 0.05, "professional": 0.04, "enterprise": 0.03, "custom": 0.02},
        "value_token": {"basic": 0.08, "professional": 0.06, "enterprise": 0.04, "custom": 0.03},
        "value_batch": {"basic": 50, "professional": 40, "enterprise": 30, "custom": 20},
        "asset_summary": {"basic": 50, "professional": 40, "enterprise": 30, "custom": 20},
        "audit_trace": {"basic": 20, "professional": 15, "enterprise": 0, "custom": 0},
        "research": {"basic": 100, "professional": 80, "enterprise": 50, "custom": 30},
    }

    OVER_RATES = {
        "predict_level": 0.10,
        "predict_quality": 0.10,
        "value_token": 0.15,
        "value_batch": 100,
        "asset_summary": 100,
    }

    CREATE_USER_TABLE = """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        plan TEXT NOT NULL DEFAULT 'basic',
        api_key TEXT UNIQUE NOT NULL,
        created_at REAL NOT NULL,
        subscription_start REAL NOT NULL,
        subscription_end REAL NOT NULL,
        daily_calls INTEGER DEFAULT 0,
        last_reset REAL NOT NULL,
        training_used INTEGER DEFAULT 0,
        extra_json TEXT
    );
    """

    CREATE_CALL_LOG_TABLE = """
    CREATE TABLE IF NOT EXISTS call_logs (
        log_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        api_key TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        call_count INTEGER DEFAULT 1,
        call_timestamp REAL NOT NULL,
        cost REAL DEFAULT 0.0,
        success INTEGER DEFAULT 1
    );
    """

    INDEX_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
        "CREATE INDEX IF NOT EXISTS idx_call_logs_user ON call_logs(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_call_logs_timestamp ON call_logs(call_timestamp);",
    ]

    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        self.config = _load_config(config_path)
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "outputs", "auth.db")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(self.CREATE_USER_TABLE)
            conn.execute(self.CREATE_CALL_LOG_TABLE)
            for sql in self.INDEX_SQL:
                conn.execute(sql)
            conn.commit()

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _generate_api_key(self) -> str:
        return "HAI-" + str(uuid.uuid4()).replace("-", "").upper()[:24]

    def _reset_daily_calls(self, user_id: str):
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT last_reset FROM users WHERE user_id=?", (user_id,)
            )
            row = cur.fetchone()
            if row:
                last_reset = float(row[0])
                if now - last_reset >= 24 * 3600:
                    conn.execute(
                        "UPDATE users SET daily_calls=0, last_reset=? WHERE user_id=?",
                        (now, user_id),
                    )
                    conn.commit()

    # ---------------------------
    # 用户注册
    # ---------------------------
    def register(
        self,
        username: str,
        email: str,
        password: str,
        plan: str = "basic",
        subscription_days: int = 365,
    ) -> Dict:
        if plan not in self.PLANS:
            raise ValueError(f"无效套餐: {plan}")

        user_id = str(uuid.uuid4())
        api_key = self._generate_api_key()
        now = time.time()
        subscription_end = now + subscription_days * 24 * 3600

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        user_id, username, email, password_hash, plan, api_key,
                        created_at, subscription_start, subscription_end,
                        daily_calls, last_reset, training_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        username,
                        email,
                        self._hash_password(password),
                        plan,
                        api_key,
                        now,
                        now,
                        subscription_end,
                        0,
                        now,
                        0,
                    ),
                )
                conn.commit()
            return {"user_id": user_id, "api_key": api_key, "plan": plan}
        except sqlite3.IntegrityError:
            raise ValueError("用户名或邮箱已存在")

    # ---------------------------
    # 用户登录
    # ---------------------------
    def login(self, email: str, password: str) -> Optional[Dict]:
        password_hash = self._hash_password(password)
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT user_id, username, email, plan, api_key, subscription_end
                FROM users WHERE email=? AND password_hash=?
                """,
                (email, password_hash),
            )
            row = cur.fetchone()
            if not row:
                return None
            now = time.time()
            if float(row["subscription_end"]) < now:
                raise ValueError("订阅已过期，请续费")
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "email": row["email"],
                "plan": row["plan"],
                "api_key": row["api_key"],
                "subscription_end": row["subscription_end"],
            }

    # ---------------------------
    # API 密钥验证
    # ---------------------------
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT user_id, username, plan, subscription_end, daily_calls, last_reset
                FROM users WHERE api_key=?
                """,
                (api_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            now = time.time()
            if float(row["subscription_end"]) < now:
                return None
            self._reset_daily_calls(row["user_id"])
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "plan": row["plan"],
                "subscription_end": row["subscription_end"],
                "daily_calls": int(row["daily_calls"]),
            }

    # ---------------------------
    # 检查套餐权限
    # ---------------------------
    def check_permission(self, api_key: str, endpoint: str, call_count: int = 1) -> Tuple[bool, Dict]:
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return False, {"error": "无效的 API 密钥"}

        plan = user_info["plan"]
        plan_config = self.PLANS.get(plan, self.PLANS["basic"])
        daily_limit = plan_config["daily_limit"]

        if daily_limit > 0 and user_info["daily_calls"] + call_count > daily_limit:
            return False, {
                "error": "超出每日调用限制",
                "daily_calls": user_info["daily_calls"],
                "daily_limit": daily_limit,
            }

        return True, {
            "user_id": user_info["user_id"],
            "username": user_info["username"],
            "plan": plan,
            "plan_name": plan_config["name"],
            "daily_calls": user_info["daily_calls"],
            "daily_limit": daily_limit,
        }

    # ---------------------------
    # 记录 API 调用
    # ---------------------------
    def record_call(
        self,
        api_key: str,
        endpoint: str,
        call_count: int = 1,
        success: bool = True,
    ) -> float:
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return 0.0

        plan = user_info["plan"]
        base_rate = self.API_RATES.get(endpoint, {}).get(plan, 0.0)
        cost = base_rate * call_count

        now = time.time()
        log_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO call_logs (
                    log_id, user_id, api_key, endpoint, call_count,
                    call_timestamp, cost, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    user_info["user_id"],
                    api_key,
                    endpoint,
                    call_count,
                    now,
                    cost,
                    1 if success else 0,
                ),
            )
            conn.execute(
                "UPDATE users SET daily_calls=daily_calls+? WHERE api_key=?",
                (call_count, api_key),
            )
            conn.commit()

        return cost

    # ---------------------------
    # 获取用户信息
    # ---------------------------
    def get_user_info(self, api_key: str) -> Optional[Dict]:
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return None

        plan = user_info["plan"]
        plan_config = self.PLANS.get(plan, self.PLANS["basic"])

        return {
            "user_id": user_info["user_id"],
            "username": user_info["username"],
            "plan": plan,
            "plan_name": plan_config["name"],
            "price": plan_config["price"],
            "daily_limit": plan_config["daily_limit"],
            "max_tokens": plan_config["max_tokens"],
            "allowed_categories": plan_config["allowed_categories"],
            "allowed_data_types": plan_config["allowed_data_types"],
            "support_hours": plan_config["support_hours"],
            "report_frequency": plan_config["report_frequency"],
            "training_count": plan_config["training_count"],
            "subscription_end": user_info["subscription_end"],
            "daily_calls": user_info["daily_calls"],
        }

    # ---------------------------
    # 更新用户套餐
    # ---------------------------
    def update_plan(self, api_key: str, new_plan: str, subscription_days: int = 365) -> bool:
        if new_plan not in self.PLANS:
            raise ValueError(f"无效套餐: {new_plan}")

        user_info = self.validate_api_key(api_key)
        if not user_info:
            return False

        now = time.time()
        subscription_end = now + subscription_days * 24 * 3600

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users SET plan=?, subscription_start=?, subscription_end=?
                WHERE api_key=?
                """,
                (new_plan, now, subscription_end, api_key),
            )
            conn.commit()

        return True

    # ---------------------------
    # 统计信息
    # ---------------------------
    def get_user_stats(self, api_key: str, days: int = 7) -> Dict:
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return {}

        since = time.time() - days * 24 * 3600
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT endpoint, SUM(call_count) as total_calls, SUM(cost) as total_cost
                FROM call_logs WHERE user_id=? AND call_timestamp>=?
                GROUP BY endpoint
                """,
                (user_info["user_id"], since),
            )
            by_endpoint = {row["endpoint"]: {"calls": int(row["total_calls"]), "cost": float(row["total_cost"])}
                          for row in cur.fetchall()}

            cur = conn.execute(
                """
                SELECT SUM(call_count) as total_calls, SUM(cost) as total_cost
                FROM call_logs WHERE user_id=? AND call_timestamp>=?
                """,
                (user_info["user_id"], since),
            )
            total = cur.fetchone()

        return {
            "user_id": user_info["user_id"],
            "plan": user_info["plan"],
            "days": days,
            "total_calls": int(total["total_calls"]) if total else 0,
            "total_cost": round(float(total["total_cost"]) if total else 0.0, 2),
            "by_endpoint": by_endpoint,
        }

    # ---------------------------
    # 检查科室权限
    # ---------------------------
    def check_category_permission(self, api_key: str, category: str) -> bool:
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return False

        plan = user_info["plan"]
        plan_config = self.PLANS.get(plan, self.PLANS["basic"])
        allowed_categories = plan_config["allowed_categories"]

        if plan == "custom":
            return True
        if category.lower() in allowed_categories:
            return True
        return False

    def __repr__(self) -> str:
        return f"AuthManager(db={self.db_path})"
