# -*- coding: utf-8 -*-
"""
api / admin_api.py
管理后台接口

提供管理后台功能：
1. 用户管理
2. 报告管理
3. 培训管理
4. 系统统计
"""

import os
import sys

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from modules.auth_manager import AuthManager
from modules.report_generator import ReportGenerator
from modules.training_manager import TrainingManager
from modules.audit_trail import AuditTrailModule

app = FastAPI(title="Healthcare AI Admin API", version="1.0.0")

ADMIN_API_KEY = "HAI_ADMIN_2026_SECRET_KEY"
admin_api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def get_admin(x_admin_key: str = Depends(admin_api_key_header)) -> bool:
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="无管理员权限")
    return True


auth = AuthManager()
report_gen = ReportGenerator()
training = TrainingManager()
audit = AuditTrailModule()


class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    plan: Optional[str] = "basic"


class UpdateUserRequest(BaseModel):
    plan: Optional[str] = None
    subscription_days: Optional[int] = None


# -----------------------------
# 用户管理
# -----------------------------
@app.get("/admin/users", summary="获取所有用户")
def get_users(admin: bool = Depends(get_admin)):
    with auth._connect() as conn:
        cur = conn.execute(
            "SELECT user_id, username, email, plan, api_key, created_at, subscription_end, daily_calls FROM users"
        )
        users = []
        for row in cur.fetchall():
            users.append({
                "user_id": row["user_id"],
                "username": row["username"],
                "email": row["email"],
                "plan": row["plan"],
                "plan_name": auth.PLANS.get(row["plan"], {}).get("name", row["plan"]),
                "api_key": row["api_key"],
                "created_at": row["created_at"],
                "subscription_end": row["subscription_end"],
                "daily_calls": row["daily_calls"],
            })
    return {"code": 200, "message": "success", "data": users}


@app.post("/admin/users", summary="创建用户")
def create_user(req: CreateUserRequest, admin: bool = Depends(get_admin)):
    try:
        result = auth.register(
            username=req.username,
            email=req.email,
            password=req.password,
            plan=req.plan,
        )
        return {"code": 200, "message": "创建成功", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/users/{user_id}", summary="获取用户详情")
def get_user(user_id: str, admin: bool = Depends(get_admin)):
    with auth._connect() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE user_id=?", (user_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="用户不存在")
        user_info = dict(row)
        user_info["plan_name"] = auth.PLANS.get(user_info["plan"], {}).get("name", user_info["plan"])
    return {"code": 200, "message": "success", "data": user_info}


@app.put("/admin/users/{user_id}", summary="更新用户")
def update_user(user_id: str, req: UpdateUserRequest, admin: bool = Depends(get_admin)):
    with auth._connect() as conn:
        cur = conn.execute("SELECT api_key FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="用户不存在")
        api_key = row["api_key"]

    if req.plan:
        auth.update_plan(api_key, req.plan, req.subscription_days or 365)

    return {"code": 200, "message": "更新成功"}


@app.delete("/admin/users/{user_id}", summary="删除用户")
def delete_user(user_id: str, admin: bool = Depends(get_admin)):
    with auth._connect() as conn:
        conn.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM call_logs WHERE user_id=?", (user_id,))
        conn.commit()
    return {"code": 200, "message": "删除成功"}


# -----------------------------
# 报告管理
# -----------------------------
@app.get("/admin/reports", summary="获取报告列表")
def get_reports(admin: bool = Depends(get_admin)):
    reports = report_gen.list_reports()
    return {"code": 200, "message": "success", "data": reports}


@app.get("/admin/reports/{filename}", summary="获取报告内容")
def get_report(filename: str, admin: bool = Depends(get_admin)):
    content = report_gen.get_report(filename)
    if not content:
        raise HTTPException(status_code=404, detail="报告不存在")
    return {"code": 200, "message": "success", "data": {"filename": filename, "content": content}}


@app.post("/admin/reports/quality", summary="生成质量报告")
def generate_quality_report(period: str = "monthly", admin: bool = Depends(get_admin)):
    content = report_gen.generate_quality_report(period)
    return {"code": 200, "message": "生成成功", "data": {"period": period, "content": content}}


@app.post("/admin/reports/research", summary="生成研究报告")
def generate_research_report(category: Optional[str] = None, sample: int = 10000, admin: bool = Depends(get_admin)):
    content = report_gen.generate_research_report(category, sample)
    return {"code": 200, "message": "生成成功", "data": {"category": category, "content": content}}


@app.post("/admin/reports/audit", summary="生成审计报告")
def generate_audit_report(days: int = 30, admin: bool = Depends(get_admin)):
    content = report_gen.generate_audit_report(days)
    return {"code": 200, "message": "生成成功", "data": {"days": days, "content": content}}


# -----------------------------
# 培训管理
# -----------------------------
@app.get("/admin/training/courses", summary="获取课程列表")
def get_courses(admin: bool = Depends(get_admin)):
    courses = training.get_courses()
    return {"code": 200, "message": "success", "data": courses}


@app.get("/admin/training/schedules", summary="获取培训日程")
def get_schedules(course_id: Optional[str] = None, admin: bool = Depends(get_admin)):
    schedules = training.get_schedules(course_id)
    return {"code": 200, "message": "success", "data": schedules}


@app.get("/admin/training/enrollments", summary="获取所有报名记录")
def get_enrollments(admin: bool = Depends(get_admin)):
    with training._connect() as conn:
        cur = conn.execute(
            """
            SELECT e.*, c.course_name, s.scheduled_date, s.trainer
            FROM training_enrollments e
            JOIN training_schedules s ON e.schedule_id = s.schedule_id
            JOIN training_courses c ON s.course_id = c.course_id
            ORDER BY e.enrolled_at DESC
            """
        )
        enrollments = [dict(row) for row in cur.fetchall()]
    return {"code": 200, "message": "success", "data": enrollments}


@app.get("/admin/training/stats", summary="获取培训统计")
def get_training_stats(days: int = 30, admin: bool = Depends(get_admin)):
    stats = training.get_training_stats(days)
    return {"code": 200, "message": "success", "data": stats}


# -----------------------------
# 系统统计
# -----------------------------
@app.get("/admin/stats", summary="获取系统统计")
def get_system_stats(admin: bool = Depends(get_admin)):
    with auth._connect() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_calls = conn.execute("SELECT COUNT(*) FROM call_logs").fetchone()[0]

    audit_stats = audit.statistics(days=30)
    training_stats = training.get_training_stats(days=30)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "total_users": int(total_users),
            "total_api_calls": int(total_calls),
            "audit": audit_stats,
            "training": training_stats,
        },
    }


# -----------------------------
# 审计管理
# -----------------------------
@app.get("/admin/audit/logs", summary="获取审计日志")
def get_audit_logs(days: int = 7, limit: int = 100, admin: bool = Depends(get_admin)):
    since = time.time() - days * 24 * 3600
    with audit._connect() as conn:
        cur = conn.execute(
            "SELECT * FROM audit_log WHERE access_timestamp >= ? ORDER BY access_timestamp DESC LIMIT ?",
            (since, limit),
        )
        logs = [dict(row) for row in cur.fetchall()]
    return {"code": 200, "message": "success", "data": logs}


@app.get("/admin/audit/anomalies", summary="获取异常检测")
def get_anomalies(minutes: int = 60, admin: bool = Depends(get_admin)):
    anomalies = audit.detect_anomalies(minutes)
    return {"code": 200, "message": "success", "data": anomalies}


@app.get("/admin/audit/high-risk", summary="获取高风险日志")
def get_high_risk(limit: int = 100, admin: bool = Depends(get_admin)):
    logs = audit.high_risk_logs(limit)
    return {"code": 200, "message": "success", "data": logs}


import time

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
