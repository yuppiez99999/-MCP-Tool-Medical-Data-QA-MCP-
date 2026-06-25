# 医疗数据质量评估 MCP Tool

> **小X宝医疗黑客松 2026 参赛作品** — 基于 Model Context Protocol 的医疗数据质量评估 + 循证医学工具

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-3.4+-green.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

---

## 📖 项目简介

**医疗数据质量评估 MCP Tool** 是一个基于 Model Context Protocol (MCP) 标准的医疗数据质量评估 + 循证医学服务，为医疗 AI 模型训练、数据资产入表、合规审计等场景提供端到端的数据质量评估能力。

### 核心价值

- 🔍 **真实数据接入**: 5,000,000 条北数所合规医疗 Token 数据
- ⚡ **毫秒级响应**: 50,000 条分层采样 + Parquet 缓存
- 🎯 **8 大 MCP 工具**: 覆盖数据质量评估全流程
- 🏥 **8 大科室分类**: 放射科/病理科/神经内科/心血管科/检验科/骨科/儿科/急诊科
- 📊 **4 维度评分**: 完整性 + 准确性 + 时效性 + 合规性
- 📚 **循证医学集成**: KnowS 医学文献检索 + 循证质量报告
- 🤖 **文献驱动分类**: 基于 480 篇医学文献训练的 TF-IDF + LR 科室分类模型
- 🖥️ **Gradio Web UI**: 可视化界面，支持 ModelScope Studio 部署

---

## 🛠️ MCP 工具列表

| 工具名称 | 功能描述 | 输入 | 输出 |
|---------|---------|------|------|
| `get_dataset_stats` | 获取真实数据集统计信息 | 无 | 数据集规模/等级分布/科室分布 |
| `sample_real_records` | 采样真实数据记录 | department, n | 真实 Token 记录列表 |
| `assess_data_quality` | 评估医疗数据质量 | records, department | 4 维度评分 + A/B/C/D 等级 |
| `classify_department` | 自动科室分类（ML+规则双引擎） | record | 科室 + 置信度 + 关键词解释 |
| `grade_data_level` | 数据等级评定 | quality_score | A/B/C/D 等级 |
| `generate_quality_report` | 生成完整质量报告 | records | Markdown 报告 |
| `search_medical_evidence` | 循证医学文献检索（KnowS） | query, source | 中英文文献 + 临床指南 |
| `generate_evidence_based_report` | 生成带循证引用的质量报告 | records, department | 质量报告 + 文献引用 + 改进建议 |
| `search_similar_data` | 检索相似数据 | record, top_k | 相似记录 + 相似度 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- FastMCP 3.4+
- Pandas, NumPy, PyArrow, scikit-learn

### 安装

```bash
git clone https://github.com/yuppiez99999/-MCP-Tool-Medical-Data-QA-MCP-.git
cd -MCP-Tool-Medical-Data-QA-MCP-
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并填入：

```env
KNOWS_API_KEY=sk-knows-xxx          # KnowS 循证医学 API Key
```

### 启动 MCP Server

```bash
python mcp_server.py
```

Server 将通过 stdio 协议启动，等待 MCP 客户端连接。

### 启动 Gradio Web 界面

```bash
python app.py
```

访问 `http://localhost:7860` 打开 Web 界面。

---

## 🔗 MCP 客户端配置

### Claude Desktop

编辑 `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "medical-data-qa": {
      "command": "python",
      "args": ["e:/各种PY程序/18-医疗AI模型系统/mcp_server.py"]
    }
  }
}
```

### Cursor / VS Code

在 MCP 配置中添加:

```json
{
  "mcpServers": {
    "medical-data-qa": {
      "command": "python",
      "args": ["./mcp_server.py"],
      "cwd": "./18-医疗AI模型系统"
    }
  }
}
```

### ModelScope MCP 广场

已在 ModelScope MCP 广场上线，可直接订阅使用。

**地址**: https://modelscope.cn/mcp/servers/yuppiez/leo

---

## 📚 循证医学集成（KnowS）

### 功能特点

- **多源检索**: 支持英文论文、中文论文、临床指南
- **智能关联**: 自动识别数据质量薄弱维度，检索相关医学文献
- **循证报告**: 质量报告自动附带文献引用和改进建议
- **文献驱动**: 科室分类模型基于真实医学文献训练

### 支持的数据源

| 数据源 | 标识 | 说明 |
|--------|------|------|
| 英文论文 | `paper_en` | PubMed / PMC 英文文献 |
| 中文论文 | `paper_cn` | 中文医学期刊论文 |
| 临床指南 | `guide` | 国内外临床诊疗指南 |

### 训练科室分类模型

```bash
python scripts/train_department_classifier.py
```

从 KnowS API 检索 8 个科室各 60 篇文献（共 480 篇），训练 TF-IDF + Logistic Regression 分类模型。

**模型性能**: 测试准确率 **80.0%**（8/10 测试用例）

---

## 📊 使用示例

### 1. 获取数据集统计

```python
# MCP 工具调用
result = get_dataset_stats()
# 返回: 5,000,000 条 / A级416万 + B级83万 / 8大科室
```

### 2. 评估数据质量

```python
records = [
    {
        "token_id": "HEALTH-001",
        "category": "radiology",
        "data_type": "ct_image",
        "completeness": 98.5,
        "accuracy": 97.0,
        "timeliness": 95.0,
        "compliance_score": 100.0
    }
]
result = assess_data_quality(records, department="radiology")
# 返回: 综合评分 97.6 / A级 / 4维度明细
```

### 3. 循证医学文献检索

```python
result = search_medical_evidence(
    query="CT影像质量控制",
    source="paper_cn"
)
# 返回: 相关中文文献列表 + 标题 + 摘要 + 来源
```

### 4. 生成带循证引用的质量报告

```python
result = generate_evidence_based_report(
    records=records,
    department="radiology"
)
# 返回: 质量报告 + 薄弱维度改进建议 + 文献引用
```

### 5. 文献驱动科室分类

```python
record = {
    "text": "CT scan of brain showing acute ischemic stroke",
    "data_type": "image"
}
result = classify_department(record)
# 返回: 放射科 / 置信度 24.65% / 关键词: ct, brain tumor, brain
# classification_method: ml_model (优先ML，回退规则)
```

---

## 🏥 数据集说明

### 数据来源

- **数据集**: 北数所 A 级 / B 级医疗健康 Token 数据集
- **规模**: 5,000,000 条 Token 记录
- **合规**: 严格遵循《数据安全法》《个人信息保护法》

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| token_id | string | Token 唯一标识 |
| category | string | 医疗科室 (8 类) |
| data_type | string | 数据形态 (ct_image/blood_test/ecg 等) |
| data_quality_score | float | 综合质量分 (0-100) |
| token_level | string | 等级 (A/B) |
| completeness | float | 完整性分 |
| accuracy | float | 准确性分 |
| timeliness | float | 时效性分 |
| compliance_score | float | 合规性分 |

### 8 大医疗科室

| 科室 | 英文 | 数据类型 | 基准价值 |
|------|------|---------|---------|
| 放射科 | radiology | ct_image | ¥15.00 |
| 病理科 | pathology | pathology_slide | ¥14.00 |
| 神经内科 | neurology | eeg | ¥13.50 |
| 心血管科 | cardiology | ecg | ¥13.00 |
| 检验科 | laboratory | blood_test | ¥10.00 |
| 骨科 | orthopedics | x_ray | ¥9.50 |
| 儿科 | pediatrics | growth_record | ¥9.00 |
| 急诊科 | emergency | triage | ¥8.50 |

---

## 📈 质量评估算法

### 4 维度加权评分

```
综合评分 = 完整性 × 30% + 准确性 × 35% + 时效性 × 15% + 合规性 × 20%
```

### 等级评定标准

| 等级 | 分数范围 | 说明 |
|------|---------|------|
| A | ≥ 90 | 高质量数据，可用于 AI 模型训练 |
| B | 75-89 | 标准质量数据，可用于一般分析 |
| C | 60-74 | 合格数据，需进一步处理 |
| D | < 60 | 不合格数据，不建议使用 |

---

## 🤖 文献驱动科室分类

### 双引擎架构

```
输入文本
    ↓
┌─────────────┐    文本充足?    ┌──────────────┐
│  ML 模型    │─────是──────→│ TF-IDF + LR  │
│  (优先)     │              │  8科室分类   │
└─────────────┘              └──────────────┘
       ↓否
┌─────────────┐
│  规则引擎    │
│  (回退)      │
└─────────────┘
```

### 模型特点

- **训练数据**: KnowS API 检索 480 篇医学文献（8科室 × 60篇）
- **模型算法**: TF-IDF 特征 + Logistic Regression
- **置信度输出**: 概率分布 + Top 3 备选科室
- **可解释性**: Top 5 关键词解释分类依据
- **优雅降级**: 文本不足时自动回退到规则引擎

---

## 🧪 测试

```bash
python test_real_data.py
```

---

## 📁 项目结构

```
18-医疗AI模型系统/
├── mcp_server.py               # MCP Server 入口 (FastMCP + stdio)
├── app.py                       # Gradio Web 界面 (8个Tab)
├── modelscope.json               # ModelScope 部署配置
├── config.yaml                 # 全局配置
├── requirements.txt            # Python 依赖
├── test_real_data.py           # 集成测试
├── .env                        # 环境变量 (KNOWS_API_KEY 等)
├── scripts/
│   └── train_department_classifier.py  # 文献驱动科室分类模型训练
├── data/
│   └── loader.py               # 真实数据加载器 (分层采样+Parquet)
├── models/
│   └── classifier.py            # 多任务学习分类器
├── modules/
│   ├── audit_trail.py           # 合规审计
│   ├── data_exchange.py         # 北数所登记
│   └── ...
├── api/
│   └── healthcare_ai_extension.py
├── knows-evidence-search/       # KnowS 循证医学检索工具
└── outputs/                     # 运行时生成 (git忽略)
    ├── data_sample.parquet      # 50,000条采样缓存
    ├── data_stats.json          # 统计信息
    └── department_classifier.pkl  # 科室分类模型
```

---

## 🎯 应用场景

1. **医疗 AI 模型训练** — 筛选高质量 Token 用于模型训练
2. **数据资产入表** — 为医疗机构数据资产估值提供定价基准
3. **合规审计溯源** — 通过 token_id 实现全链路追踪
4. **多模态医学研究** — 跨科室 Token 聚合分析
5. **北数所数据产品登记** — 符合数据资产登记规范
6. **循证医学辅助** — 质量改进建议附带医学文献支持
7. **科室自动分类** — 基于文本内容智能识别数据所属科室

---

## 🏆 参赛信息

- **比赛**: 小X宝医疗黑客松 2026
- **赛道**: 医疗垂直领域 MCP 工具/Skill 开发
- **阶段**: 阶段1 选题登记 (6/24截止) ✅
- **作者**: yuppiez99999

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 🤝 贡献

**作者**: yuppiez99999

小X宝医疗黑客松 2026 参赛作品

---

## 📞 联系

- GitHub: [@yuppiez99999](https://github.com/yuppiez99999)
- ModelScope MCP: [医疗数据质量评估 MCP Tool](https://modelscope.cn/mcp/servers/yuppiez/leo)
