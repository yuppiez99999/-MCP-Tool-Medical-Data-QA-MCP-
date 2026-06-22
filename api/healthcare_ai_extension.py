# -*- coding: utf-8 -*-
"""
api / healthcare_ai_extension.py
医疗 AI 模型系统专用 FastAPI 服务（完整版）

提供完整的产品功能：
1. 用户认证与套餐管理
2. Token 等级预测、质量回归、科室分类
3. 数据资产估值（单条/批量）
4. 合规审计溯源
5. 医学研究分析
6. 北数所登记报告生成
7. WebSocket 实时推送
8. 报告管理与培训管理

使用方法：
  cd 18-医疗AI模型系统
  python api/healthcare_ai_extension.py --port 8001

API 文档：
  http://localhost:8001/docs
  http://localhost:8001/redoc
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

import yaml
import numpy as np
import pandas as pd

from data.loader import HealthcareTokenLoader
from data.preprocessor import HealthcareTokenPreprocessor
from models.valuation_engine import AssetValuationEngine
from models.research_framework import MultimodalResearchFramework
from modules.audit_trail import AuditTrailModule
from modules.data_exchange import DataExchangeRegistrar
from modules.auth_manager import AuthManager
from modules.report_generator import ReportGenerator
from modules.training_manager import TrainingManager

app = FastAPI(
    title="Healthcare AI Model System",
    version="1.0.0",
    description="医疗健康数据 AI 模型系统 - 提供 Token 分类、资产估值、合规审计等完整功能",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# -----------------------------
# 全局配置与加载
# -----------------------------
def _load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = _load_config(os.path.join(BASE_DIR, "config.yaml"))
CATEGORIES = list(CONFIG["categories"].keys())
DATA_TYPES = list(CONFIG["data_types"].keys())

LOADER = HealthcareTokenLoader(config_path=os.path.join(BASE_DIR, "config.yaml"))
PREPROCESSOR = HealthcareTokenPreprocessor(categories=CATEGORIES, data_types=DATA_TYPES)
AUDIT = AuditTrailModule(config=CONFIG.get("audit", {}))
AUTH = AuthManager(config_path=os.path.join(BASE_DIR, "config.yaml"))
REPORT_GEN = ReportGenerator(config_path=os.path.join(BASE_DIR, "config.yaml"))
TRAINING = TrainingManager(config_path=os.path.join(BASE_DIR, "config.yaml"))

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        model_path = os.path.join(BASE_DIR, "outputs", "healthcare_token_classifier.pt")
        if os.path.exists(model_path):
            from models.classifier import HealthcareTokenClassifier
            _classifier = HealthcareTokenClassifier.load(model_path)
    return _classifier


# -----------------------------
# 请求模型
# -----------------------------
class TokenRow(BaseModel):
    token_id: str
    category: str
    data_type: str
    token_level: Optional[str] = "A"
    data_quality_score: float = 98.0
    completeness: float = 99.0
    accuracy: float = 97.0
    timeliness: float = 95.0
    compliance_score: float = 100.0
    entity_id: Optional[str] = None


class TokenBatchRequest(BaseModel):
    tokens: List[TokenRow]


class AuditLogRequest(BaseModel):
    token_id: str
    access_type: str = Field(..., description="QUERY / TRAIN / EXPORT / ANALYSIS / REGISTRATION")
    accessor_id: str
    access_purpose: str
    entity_id: Optional[str] = None
    compliance_score: Optional[float] = None


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    plan: Optional[str] = "basic"


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdatePlanRequest(BaseModel):
    new_plan: str
    subscription_days: Optional[int] = 365


class EnrollRequest(BaseModel):
    schedule_id: str


# -----------------------------
# 认证接口
# -----------------------------
@app.post("/api/auth/register", summary="用户注册")
def register(req: RegisterRequest):
    try:
        result = AUTH.register(
            username=req.username,
            email=req.email,
            password=req.password,
            plan=req.plan,
        )
        return {"code": 200, "message": "注册成功", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login", summary="用户登录")
def login(req: LoginRequest):
    try:
        result = AUTH.login(email=req.email, password=req.password)
        if not result:
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        return {"code": 200, "message": "登录成功", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/auth/info", summary="获取用户信息")
def get_user_info(x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.get_user_info(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")
    return {"code": 200, "message": "success", "data": user_info}


@app.post("/api/auth/plan", summary="更新用户套餐")
def update_plan(req: UpdatePlanRequest, x_api_key: str = Depends(api_key_header)):
    success = AUTH.update_plan(
        api_key=x_api_key,
        new_plan=req.new_plan,
        subscription_days=req.subscription_days,
    )
    if not success:
        raise HTTPException(status_code=400, detail="更新套餐失败")
    return {"code": 200, "message": "套餐更新成功"}


@app.get("/api/auth/stats", summary="获取用户统计信息")
def get_user_stats(days: int = 7, x_api_key: str = Depends(api_key_header)):
    stats = AUTH.get_user_stats(api_key=x_api_key, days=days)
    if not stats:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")
    return {"code": 200, "message": "success", "data": stats}


@app.get("/api/auth/plans", summary="获取所有套餐信息")
def get_all_plans():
    plans = []
    for key, config in AUTH.PLANS.items():
        plans.append({
            "plan_key": key,
            "name": config["name"],
            "price": config["price"],
            "daily_limit": config["daily_limit"],
            "max_tokens": config["max_tokens"],
            "allowed_categories": config["allowed_categories"],
            "support_hours": config["support_hours"],
            "report_frequency": config["report_frequency"],
            "training_count": config["training_count"],
        })
    return {"code": 200, "message": "success", "data": plans}


# -----------------------------
# 医疗 AI 接口
# -----------------------------
@app.post("/api/healthcare/predict-level", summary="预测 Token 等级 (A/B)")
def predict_level(tokens: TokenBatchRequest, x_api_key: str = Depends(api_key_header)):
    ok, info = AUTH.check_permission(x_api_key, "predict_level", len(tokens.tokens))
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    clf = get_classifier()
    df = pd.DataFrame([t.model_dump() for t in tokens.tokens])
    X = PREPROCESSOR.transform(df)
    preds = clf.predict(X) if clf is not None else None

    if preds is None:
        level_pred = (df["data_quality_score"].values >= 97.0).astype(int).tolist()
        level_proba = [[1.0 - q / 100.0, q / 100.0] for q in df["data_quality_score"].values]
    else:
        level_pred = preds["level_pred"].tolist()
        level_proba = preds["level_proba"].tolist()

    AUTH.record_call(x_api_key, "predict_level", len(tokens.tokens))

    return {
        "code": 200,
        "message": "success",
        "data": {
            "predictions": [
                {"token_id": t.token_id, "level": "A" if level_pred[i] == 1 else "B", "proba": level_proba[i]}
                for i, t in enumerate(tokens.tokens)
            ],
            "user_info": info,
        },
    }


@app.post("/api/healthcare/predict-quality", summary="预测综合质量分")
def predict_quality(tokens: TokenBatchRequest, x_api_key: str = Depends(api_key_header)):
    ok, info = AUTH.check_permission(x_api_key, "predict_quality", len(tokens.tokens))
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    clf = get_classifier()
    df = pd.DataFrame([t.model_dump() for t in tokens.tokens])
    if clf is not None:
        X = PREPROCESSOR.transform(df)
        preds = clf.predict(X)
        quality_pred = preds["quality_pred"].tolist()
    else:
        quality_pred = [
            (t.completeness + t.accuracy + t.timeliness + t.compliance_score) / 4.0
            for t in tokens.tokens
        ]

    AUTH.record_call(x_api_key, "predict_quality", len(tokens.tokens))

    return {
        "code": 200,
        "message": "success",
        "data": {
            "predictions": [
                {"token_id": t.token_id, "quality_score": round(float(quality_pred[i]), 4)}
                for i, t in enumerate(tokens.tokens)
            ],
            "user_info": info,
        },
    }


@app.post("/api/healthcare/value-token", summary="单 Token 估值")
def value_token(token: TokenRow, x_api_key: str = Depends(api_key_header)):
    ok, info = AUTH.check_permission(x_api_key, "value_token", 1)
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    if not AUTH.check_category_permission(x_api_key, token.category):
        raise HTTPException(status_code=403, detail=f"无权访问科室: {token.category}")

    engine = AssetValuationEngine(config_path=os.path.join(BASE_DIR, "config.yaml"))
    result = engine.value_token(token.model_dump())

    AUTH.record_call(x_api_key, "value_token", 1)

    return {"code": 200, "message": "success", "data": result}


@app.post("/api/healthcare/value-batch", summary="批量 Token 估值")
def value_batch(tokens: TokenBatchRequest, x_api_key: str = Depends(api_key_header)):
    ok, info = AUTH.check_permission(x_api_key, "value_batch", 1)
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    for token in tokens.tokens:
        if not AUTH.check_category_permission(x_api_key, token.category):
            raise HTTPException(status_code=403, detail=f"无权访问科室: {token.category}")

    engine = AssetValuationEngine(config_path=os.path.join(BASE_DIR, "config.yaml"))
    df = pd.DataFrame([t.model_dump() for t in tokens.tokens])
    valued = engine.value_dataframe(df)
    summary = engine.summary(valued)

    AUTH.record_call(x_api_key, "value_batch", 1)

    return {"code": 200, "message": "success", "data": {"detail": valued.to_dict(orient="records"), "summary": summary}}


@app.get("/api/healthcare/asset-summary", summary="数据资产全景摘要")
def asset_summary(x_api_key: str = Depends(api_key_header)):
    ok, info = AUTH.check_permission(x_api_key, "asset_summary", 1)
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    summary = LOADER.summary()
    engine = AssetValuationEngine(
        config_path=os.path.join(BASE_DIR, "config.yaml"),
        category_counts=summary.get("category_counts", {}),
    )
    sample = LOADER.load_all(sample=10000)
    if sample.empty:
        return {"code": 200, "message": "success", "data": {"dataset": summary, "valuation": None}}
    valued = engine.value_dataframe(sample)
    val_summary = engine.summary(valued)

    AUTH.record_call(x_api_key, "asset_summary", 1)

    return {"code": 200, "message": "success", "data": {"dataset": summary, "valuation": val_summary}}


# -----------------------------
# 审计接口
# -----------------------------
@app.post("/api/healthcare/audit-log", summary="记录审计日志")
def write_audit_log(req: AuditLogRequest, x_api_key: str = Depends(api_key_header)):
    try:
        audit_id = AUDIT.log_access(
            token_id=req.token_id,
            access_type=req.access_type.upper(),
            accessor_id=req.accessor_id,
            access_purpose=req.access_purpose,
            entity_id=req.entity_id,
            compliance_score=req.compliance_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"code": 200, "message": "success", "data": {"audit_id": audit_id}}


@app.get("/api/healthcare/audit-trace/{token_id}", summary="按 Token 溯源访问历史")
def audit_trace(token_id: str, limit: int = 50, x_api_key: str = Depends(api_key_header)):
    ok, info = AUTH.check_permission(x_api_key, "audit_trace", 1)
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    logs = AUDIT.trace_token(token_id=token_id, limit=limit)
    AUTH.record_call(x_api_key, "audit_trace", 1)

    return {"code": 200, "message": "success", "data": {"token_id": token_id, "logs": logs}}


@app.get("/api/healthcare/audit/anomalies", summary="异常访问检测")
def audit_anomalies(minutes: int = 10, x_api_key: str = Depends(api_key_header)):
    return {"code": 200, "message": "success", "data": {"anomalies": AUDIT.detect_anomalies(minutes=minutes)}}


@app.get("/api/healthcare/audit/stats", summary="审计统计")
def audit_stats(days: int = 7, x_api_key: str = Depends(api_key_header)):
    return {"code": 200, "message": "success", "data": AUDIT.statistics(days=days)}


# -----------------------------
# 研究接口
# -----------------------------
@app.get("/api/healthcare/research/{category}", summary="科室研究报告")
def research_category(
    category: Optional[str] = None,
    sample: int = 10000,
    x_api_key: str = Depends(api_key_header),
):
    ok, info = AUTH.check_permission(x_api_key, "research", 1)
    if not ok:
        raise HTTPException(status_code=429, detail=info["error"])

    if category and not AUTH.check_category_permission(x_api_key, category):
        raise HTTPException(status_code=403, detail=f"无权访问科室: {category}")

    df = LOADER.filter_by(category=category, limit=sample)
    if df.empty:
        return {"code": 200, "message": "success", "data": {"category": category, "total_tokens": 0}}

    framework = MultimodalResearchFramework(df=df, categories=CATEGORIES)
    AUTH.record_call(x_api_key, "research", 1)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "category": category,
            "report": framework.research_report(category=category),
            "quality_by_category": framework.quality_by_category().to_dict(orient="records")[:20],
        },
    }


# -----------------------------
# 登记接口
# -----------------------------
@app.post("/api/healthcare/generate-registration", summary="生成北数所登记报告")
def generate_registration(sample: int = 10000, x_api_key: str = Depends(api_key_header)):
    summary = LOADER.summary()
    df_sample = LOADER.load_all(sample=sample)
    engine = AssetValuationEngine(
        config_path=os.path.join(BASE_DIR, "config.yaml"),
        category_counts=summary.get("category_counts", {}),
    )
    valued = engine.value_dataframe(df_sample) if not df_sample.empty else pd.DataFrame()
    val_summary = engine.summary(valued)
    registrar = DataExchangeRegistrar(
        config=CONFIG,
        dataset_summary=summary,
        valuation_summary=val_summary,
    )
    md = registrar.generate_markdown(raw_df_sample=df_sample)

    return {"code": 200, "message": "success", "data": {"report_markdown": md, "summary": val_summary}}


# -----------------------------
# 报告接口
# -----------------------------
@app.get("/api/reports/list", summary="获取报告列表")
def list_reports(x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    reports = REPORT_GEN.list_reports()
    return {"code": 200, "message": "success", "data": reports}


@app.get("/api/reports/quality", summary="获取数据质量报告")
def get_quality_report(period: str = "monthly", x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    content = REPORT_GEN.generate_quality_report(period)
    return {"code": 200, "message": "success", "data": {"period": period, "content": content}}


@app.get("/api/reports/api-stats", summary="获取 API 使用统计报告")
def get_api_stats_report(days: int = 30, x_api_key: str = Depends(api_key_header)):
    content = REPORT_GEN.generate_api_stats_report(x_api_key, days)
    return {"code": 200, "message": "success", "data": {"days": days, "content": content}}


# -----------------------------
# 培训接口
# -----------------------------
@app.get("/api/training/courses", summary="获取培训课程")
def get_courses(x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    courses = TRAINING.get_courses()
    return {"code": 200, "message": "success", "data": courses}


@app.get("/api/training/schedules", summary="获取培训日程")
def get_schedules(course_id: Optional[str] = None, x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    schedules = TRAINING.get_schedules(course_id)
    return {"code": 200, "message": "success", "data": schedules}


@app.post("/api/training/enroll", summary="报名培训")
def enroll_training(req: EnrollRequest, x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    try:
        result = TRAINING.enroll(user_info["user_id"], req.schedule_id)
        return {"code": 200, "message": "报名成功", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/training/my-enrollments", summary="获取我的培训记录")
def get_my_enrollments(x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    enrollments = TRAINING.get_user_enrollments(user_info["user_id"])
    return {"code": 200, "message": "success", "data": enrollments}


@app.get("/api/training/my-certificates", summary="获取我的证书")
def get_my_certificates(x_api_key: str = Depends(api_key_header)):
    user_info = AUTH.validate_api_key(x_api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")

    certificates = TRAINING.get_user_certificates(user_info["user_id"])
    return {"code": 200, "message": "success", "data": certificates}


# -----------------------------
# WebSocket 实时推送
# -----------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/healthcare/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")

            if action == "ping":
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": time.time()},
                    client_id,
                )

            elif action == "predict_level":
                tokens_data = data.get("tokens", [])
                if not tokens_data:
                    continue

                tokens = [TokenRow(**t) for t in tokens_data]
                df = pd.DataFrame([t.model_dump() for t in tokens])
                clf = get_classifier()
                X = PREPROCESSOR.transform(df)
                preds = clf.predict(X) if clf is not None else None

                if preds is None:
                    level_pred = (df["data_quality_score"].values >= 97.0).astype(int).tolist()
                else:
                    level_pred = preds["level_pred"].tolist()

                predictions = [
                    {"token_id": tokens[i].token_id, "level": "A" if level_pred[i] == 1 else "B"}
                    for i in range(len(tokens))
                ]

                await manager.send_personal_message(
                    {"type": "predict_level_result", "predictions": predictions},
                    client_id,
                )

            elif action == "subscribe_updates":
                await manager.send_personal_message(
                    {"type": "subscribed", "message": "已订阅实时更新"},
                    client_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast({"type": "disconnected", "client_id": client_id})
    except Exception as e:
        manager.disconnect(client_id)


# -----------------------------
# 健康检查
# -----------------------------
@app.get("/health", summary="健康检查")
def health_check():
    return {"status": "ok", "system": "Healthcare AI Model System", "version": "1.0.0"}


def main():
    parser = argparse.ArgumentParser(description="Healthcare AI Extension API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
