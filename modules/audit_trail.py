# -*- coding: utf-8 -*-
"""
modules / audit_trail.py
合规审计溯源模块

核心能力：
1. 每次 Token 查询/训练/导出等操作都写入 SQLite 审计日志
2. 支持按 token_id 回溯全链路访问历史
3. 按实体 (entity_id) 聚合访问，检测异常高频访问
4. 访问者身份识别 + 访问目的声明（符合数据安全法最小必要原则）
"""

import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class AuditTrailModule:
    """合规审计溯源模块（基于 SQLite）"""

    ACCESS_TYPES = ("QUERY", "TRAIN", "EXPORT", "ANALYSIS", "REGISTRATION")

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS audit_log (
        audit_id TEXT PRIMARY KEY,
        token_id TEXT NOT NULL,
        entity_id TEXT,
        access_type TEXT NOT NULL,
        accessor_id TEXT NOT NULL,
        access_timestamp REAL NOT NULL,
        access_datetime TEXT NOT NULL,
        access_purpose TEXT,
        compliance_check INTEGER DEFAULT 1,
        risk_score REAL DEFAULT 0.0,
        extra_json TEXT
    );
    """

    INDEX_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_audit_token ON audit_log(token_id);",
        "CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_id);",
        "CREATE INDEX IF NOT EXISTS idx_audit_accessor ON audit_log(accessor_id);",
        "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(access_timestamp);",
    ]

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        # 默认 db_path 解析
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "outputs", "audit_log.db")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.config = config or {}
        self.anomaly_threshold = int(self.config.get("anomaly_threshold", 100))
        self.high_risk_threshold = float(self.config.get("high_risk_threshold", 70.0))
        self.min_purpose_length = int(self.config.get("min_purpose_length", 5))
        self._init_db()

    # ---------------------------
    # DB 初始化
    # ---------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            for sql in self.INDEX_SQL:
                conn.execute(sql)
            conn.commit()

    # ---------------------------
    # 写审计日志
    # ---------------------------
    def log_access(
        self,
        token_id: str,
        access_type: str,
        accessor_id: str,
        access_purpose: str,
        entity_id: Optional[str] = None,
        compliance_score: Optional[float] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """写入一条访问日志，返回 audit_id"""
        access_type = access_type.upper()
        if access_type not in self.ACCESS_TYPES:
            raise ValueError(f"access_type 必须是 {self.ACCESS_TYPES} 之一，当前: {access_type}")

        # 目的声明完整性校验
        if not access_purpose or len(str(access_purpose).strip()) < self.min_purpose_length:
            raise ValueError(
                f"访问目的声明至少 {self.min_purpose_length} 个字符，"
                f"当前: '{access_purpose}'"
            )

        now_ts = time.time()
        now_dt = datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S")
        audit_id = str(uuid.uuid4())

        # 合规校验：compliance_score 未给出时默认 100
        if compliance_score is None:
            compliance_check = 1
        else:
            compliance_check = 1 if compliance_score >= 95.0 else 0

        # 风险评分（简化公式：按访问类型和合规性）
        risk_score = self._calc_risk_score(
            access_type=access_type,
            compliance_check=compliance_check,
            accessor_id=accessor_id,
        )

        extra_json = "" if extra is None else _safe_json(extra)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (
                    audit_id, token_id, entity_id, access_type, accessor_id,
                    access_timestamp, access_datetime, access_purpose,
                    compliance_check, risk_score, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    token_id,
                    entity_id,
                    access_type,
                    accessor_id,
                    now_ts,
                    now_dt,
                    access_purpose,
                    int(compliance_check),
                    float(risk_score),
                    extra_json,
                ),
            )
            conn.commit()
        return audit_id

    def log_batch(self, records: List[Dict]) -> List[str]:
        """批量写入日志，records 每个元素为 log_access 的关键字参数"""
        return [self.log_access(**r) for r in records]

    # ---------------------------
    # 风险评分（简化版，可扩展为 ML 模型）
    # ---------------------------
    def _calc_risk_score(self, access_type: str, compliance_check: int, accessor_id: str) -> float:
        base = {"QUERY": 10.0, "ANALYSIS": 20.0, "TRAIN": 30.0, "EXPORT": 50.0, "REGISTRATION": 15.0}
        score = base.get(access_type, 20.0)
        if compliance_check == 0:
            score += 40.0
        # 单位时间内访问次数越多，风险越高
        recent = self._count_recent(accessor_id, minutes=10)
        if recent > self.anomaly_threshold:
            score += min(40.0, (recent - self.anomaly_threshold) * 0.5)
        return min(100.0, score)

    def _count_recent(self, accessor_id: str, minutes: int) -> int:
        since = time.time() - minutes * 60
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE accessor_id=? AND access_timestamp>=?",
                (accessor_id, since),
            )
            return int(cur.fetchone()[0])

    # ---------------------------
    # 审计溯源查询
    # ---------------------------
    def trace_token(self, token_id: str, limit: int = 100) -> List[Dict]:
        """查询某个 Token 的全链路访问历史"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM audit_log
                WHERE token_id = ?
                ORDER BY access_timestamp DESC
                LIMIT ?
                """,
                (token_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def trace_entity(self, entity_id: str, limit: int = 500) -> List[Dict]:
        """按实体查询其所有 Token 的访问记录"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM audit_log
                WHERE entity_id = ?
                ORDER BY access_timestamp DESC
                LIMIT ?
                """,
                (entity_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def trace_accessor(self, accessor_id: str, limit: int = 500) -> List[Dict]:
        """按访问者查询"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM audit_log
                WHERE accessor_id = ?
                ORDER BY access_timestamp DESC
                LIMIT ?
                """,
                (accessor_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    # ---------------------------
    # 异常检测
    # ---------------------------
    def detect_anomalies(self, minutes: int = 10) -> List[Dict]:
        """检测单位时间内的异常高频访问者"""
        since = time.time() - minutes * 60
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT accessor_id, COUNT(*) as cnt, MAX(risk_score) as max_risk
                FROM audit_log
                WHERE access_timestamp >= ?
                GROUP BY accessor_id
                HAVING cnt >= ?
                ORDER BY cnt DESC
                """,
                (since, self.anomaly_threshold),
            )
            return [dict(row) for row in cur.fetchall()]

    def high_risk_logs(self, limit: int = 100) -> List[Dict]:
        """高风险访问日志"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM audit_log
                WHERE risk_score >= ?
                ORDER BY access_timestamp DESC
                LIMIT ?
                """,
                (self.high_risk_threshold, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    # ---------------------------
    # 统计
    # ---------------------------
    def statistics(self, days: int = 7) -> Dict:
        since = time.time() - days * 24 * 3600
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE access_timestamp>=?", (since,)
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT access_type, COUNT(*) FROM audit_log WHERE access_timestamp>=? GROUP BY access_type",
                (since,),
            ).fetchall()
            high_risk = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE access_timestamp>=? AND risk_score>=?",
                (since, self.high_risk_threshold),
            ).fetchone()[0]
            return {
                "total_logs_last_n_days": int(total),
                "by_access_type": {r[0]: int(r[1]) for r in by_type},
                "high_risk_logs": int(high_risk),
                "anomaly_threshold": self.anomaly_threshold,
            }

    def __repr__(self) -> str:
        return f"AuditTrailModule(db={self.db_path})"


# -----------------------------
# JSON 辅助函数（避免强依赖 json 时也能序列化）
# -----------------------------
def _safe_json(obj: Dict) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
