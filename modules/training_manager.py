# -*- coding: utf-8 -*-
"""
modules / training_manager.py
培训管理系统

核心功能：
1. 培训课程管理
2. 培训预约与记录
3. 培训证书发放
4. 培训统计

培训类型：
- 基础操作培训（线上）：¥5,000/次（2小时）
- 高级培训（现场）：¥20,000/次（1天）
- 定制化培训：¥50,000/次（2天）
"""

import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import yaml


def _load_config(config_path: Optional[str] = None) -> Dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrainingManager:
    """培训管理系统"""

    TRAINING_TYPES = {
        "basic": {
            "name": "基础操作培训",
            "format": "线上",
            "duration": "2 小时",
            "price": 5000,
            "description": "API 基础使用、数据查询、基础分析",
        },
        "advanced": {
            "name": "高级培训",
            "format": "现场",
            "duration": "1 天",
            "price": 20000,
            "description": "深度数据分析、模型训练、定制化开发",
        },
        "custom": {
            "name": "定制化培训",
            "format": "现场",
            "duration": "2 天",
            "price": 50000,
            "description": "根据客户需求定制培训内容",
        },
    }

    CREATE_COURSE_TABLE = """
    CREATE TABLE IF NOT EXISTS training_courses (
        course_id TEXT PRIMARY KEY,
        course_name TEXT NOT NULL,
        training_type TEXT NOT NULL,
        description TEXT,
        duration TEXT,
        max_participants INTEGER DEFAULT 20,
        created_at REAL NOT NULL,
        status TEXT DEFAULT 'active'
    );
    """

    CREATE_SCHEDULE_TABLE = """
    CREATE TABLE IF NOT EXISTS training_schedules (
        schedule_id TEXT PRIMARY KEY,
        course_id TEXT NOT NULL,
        scheduled_date REAL NOT NULL,
        trainer TEXT,
        participants INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        created_at REAL NOT NULL
    );
    """

    CREATE_ENROLLMENT_TABLE = """
    CREATE TABLE IF NOT EXISTS training_enrollments (
        enrollment_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        schedule_id TEXT NOT NULL,
        status TEXT DEFAULT 'confirmed',
        attended INTEGER DEFAULT 0,
        certificate_id TEXT,
        enrolled_at REAL NOT NULL,
        completed_at REAL
    );
    """

    CREATE_CERTIFICATE_TABLE = """
    CREATE TABLE IF NOT EXISTS training_certificates (
        certificate_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        course_id TEXT NOT NULL,
        schedule_id TEXT NOT NULL,
        issued_at REAL NOT NULL,
        valid_until REAL NOT NULL
    );
    """

    INDEX_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_enrollments_user ON training_enrollments(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_enrollments_schedule ON training_enrollments(schedule_id);",
        "CREATE INDEX IF NOT EXISTS idx_certificates_user ON training_certificates(user_id);",
        "CREATE INDEX IF NOT EXISTS idx_schedules_course ON training_schedules(course_id);",
    ]

    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        self.config = _load_config(config_path)
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "outputs", "training.db")
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()
        self._init_default_courses()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(self.CREATE_COURSE_TABLE)
            conn.execute(self.CREATE_SCHEDULE_TABLE)
            conn.execute(self.CREATE_ENROLLMENT_TABLE)
            conn.execute(self.CREATE_CERTIFICATE_TABLE)
            for sql in self.INDEX_SQL:
                conn.execute(sql)
            conn.commit()

    def _init_default_courses(self):
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM training_courses")
            if cur.fetchone()[0] == 0:
                for training_type, config in self.TRAINING_TYPES.items():
                    conn.execute(
                        """
                        INSERT INTO training_courses (
                            course_id, course_name, training_type,
                            description, duration, max_participants, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"course_{training_type}",
                            config["name"],
                            training_type,
                            config["description"],
                            config["duration"],
                            20 if training_type == "basic" else 10,
                            time.time(),
                        ),
                    )
                conn.commit()

    def get_courses(self) -> List[Dict]:
        """获取所有课程"""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM training_courses WHERE status='active'")
            return [dict(row) for row in cur.fetchall()]

    def get_course(self, course_id: str) -> Optional[Dict]:
        """获取指定课程"""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM training_courses WHERE course_id=?", (course_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def create_schedule(self, course_id: str, scheduled_date: float, trainer: str) -> Dict:
        """创建培训日程"""
        schedule_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_schedules (
                    schedule_id, course_id, scheduled_date, trainer, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (schedule_id, course_id, scheduled_date, trainer, time.time()),
            )
            conn.commit()
        return {"schedule_id": schedule_id, "course_id": course_id, "scheduled_date": scheduled_date}

    def get_schedules(self, course_id: Optional[str] = None) -> List[Dict]:
        """获取培训日程"""
        with self._connect() as conn:
            if course_id:
                cur = conn.execute(
                    "SELECT * FROM training_schedules WHERE course_id=? ORDER BY scheduled_date",
                    (course_id,),
                )
            else:
                cur = conn.execute("SELECT * FROM training_schedules ORDER BY scheduled_date")
            return [dict(row) for row in cur.fetchall()]

    def enroll(self, user_id: str, schedule_id: str) -> Dict:
        """报名培训"""
        enrollment_id = str(uuid.uuid4())
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT course_id, participants, max_participants FROM training_schedules WHERE schedule_id=?",
                (schedule_id,),
            )
            schedule = cur.fetchone()
            if not schedule:
                raise ValueError("日程不存在")
            if schedule["participants"] >= schedule["max_participants"]:
                raise ValueError("名额已满")

            conn.execute(
                """
                INSERT INTO training_enrollments (
                    enrollment_id, user_id, schedule_id, enrolled_at
                ) VALUES (?, ?, ?, ?)
                """,
                (enrollment_id, user_id, schedule_id, time.time()),
            )
            conn.execute(
                "UPDATE training_schedules SET participants=participants+1 WHERE schedule_id=?",
                (schedule_id,),
            )
            conn.commit()
        return {"enrollment_id": enrollment_id, "schedule_id": schedule_id}

    def record_attendance(self, enrollment_id: str, attended: bool = True) -> bool:
        """记录出勤"""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM training_enrollments WHERE enrollment_id=?", (enrollment_id,))
            enrollment = cur.fetchone()
            if not enrollment:
                return False

            conn.execute(
                "UPDATE training_enrollments SET attended=?, completed_at=? WHERE enrollment_id=?",
                (1 if attended else 0, time.time() if attended else None, enrollment_id),
            )
            conn.commit()

            if attended:
                self._issue_certificate(enrollment["user_id"], enrollment["schedule_id"])

        return True

    def _issue_certificate(self, user_id: str, schedule_id: str):
        """发放证书"""
        certificate_id = f"CERT-{str(uuid.uuid4()).upper()[:16]}"
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT course_id FROM training_schedules WHERE schedule_id=?",
                (schedule_id,),
            )
            schedule = cur.fetchone()
            if not schedule:
                return

            conn.execute(
                """
                INSERT INTO training_certificates (
                    certificate_id, user_id, course_id, schedule_id,
                    issued_at, valid_until
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    certificate_id,
                    user_id,
                    schedule["course_id"],
                    schedule_id,
                    time.time(),
                    time.time() + 365 * 24 * 3600,
                ),
            )
            conn.commit()

    def get_user_enrollments(self, user_id: str) -> List[Dict]:
        """获取用户培训记录"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT e.*, c.course_name, c.training_type, s.scheduled_date, s.trainer
                FROM training_enrollments e
                JOIN training_schedules s ON e.schedule_id = s.schedule_id
                JOIN training_courses c ON s.course_id = c.course_id
                WHERE e.user_id = ?
                ORDER BY e.enrolled_at DESC
                """,
                (user_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_user_certificates(self, user_id: str) -> List[Dict]:
        """获取用户证书"""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT cert.*, c.course_name
                FROM training_certificates cert
                JOIN training_courses c ON cert.course_id = c.course_id
                WHERE cert.user_id = ?
                ORDER BY cert.issued_at DESC
                """,
                (user_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_training_stats(self, days: int = 30) -> Dict:
        """获取培训统计"""
        since = time.time() - days * 24 * 3600
        with self._connect() as conn:
            total_enrollments = conn.execute(
                "SELECT COUNT(*) FROM training_enrollments WHERE enrolled_at >= ?",
                (since,),
            ).fetchone()[0]

            completed = conn.execute(
                "SELECT COUNT(*) FROM training_enrollments WHERE attended=1 AND completed_at >= ?",
                (since,),
            ).fetchone()[0]

            certificates = conn.execute(
                "SELECT COUNT(*) FROM training_certificates WHERE issued_at >= ?",
                (since,),
            ).fetchone()[0]

            by_course = conn.execute(
                """
                SELECT c.course_name, COUNT(*) as cnt
                FROM training_enrollments e
                JOIN training_schedules s ON e.schedule_id = s.schedule_id
                JOIN training_courses c ON s.course_id = c.course_id
                WHERE e.enrolled_at >= ?
                GROUP BY c.course_name
                """,
                (since,),
            ).fetchall()

        return {
            "period_days": days,
            "total_enrollments": int(total_enrollments),
            "completed_training": int(completed),
            "certificates_issued": int(certificates),
            "completion_rate": round(int(completed) / max(1, int(total_enrollments)) * 100, 2),
            "by_course": {row[0]: int(row[1]) for row in by_course},
        }

    def __repr__(self) -> str:
        return f"TrainingManager(db={self.db_path})"
