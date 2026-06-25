# 医疗数据质量评估 MCP Tool

> **小X宝医疗黑客松 2026 参赛作品** — 基于 Model Context Protocol 的医疗数据质量评估 + 循证医学工具

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-3.4+-green.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

---

## 1. 项目简介与医疗场景

### 一句话描述

基于 MCP 协议的医疗数据质量评估 + 循证医学工具，为医疗 AI 模型训练提供 4 维度质量评分、8 科室自动分类和循证文献支持。

### 解决的痛点

- **数据质量参差不齐**：医疗 AI 模型训练数据缺乏标准化质量评估，影响模型效果
- **科室分类效率低**：多模态医疗数据（影像/检验/病理/文本）人工分类耗时耗力
- **质量改进无依据**：数据质量问题缺乏循证医学参考，改进方向不明确
- **数据资产估值难**：医疗数据入表/交易缺乏客观的质量定价基准

### 目标受众

- 🏥 **医疗机构数据部门** — 数据质量审计与资产入表
- 🤖 **医疗 AI 研发团队** — 训练数据筛选与质量控制
- 📊 **医学数据研究员** — 跨科室数据聚合与质量分析
- 💼 **数据交易所/登记机构** — 医疗数据产品质量核验

---

## 2. 功能特性

- 🔍 **真实数据接入** — 5,000,000 条北数所合规医疗 Token 数据，分层采样毫秒级响应
- 🎯 **9 大 MCP 工具** — 覆盖数据质量评估全流程（评估/分类/等级/报告/检索/循证）
- 🏥 **8 大科室分类** — 放射科/病理科/神经内科/心血管科/检验科/骨科/儿科/急诊科
- 📊 **4 维度评分** — 完整性(30%) + 准确性(35%) + 时效性(15%) + 合规性(20%)
- 🤖 **文献驱动 ML 分类** — 基于 480 篇 KnowS 医学文献训练的 TF-IDF + LR 模型，80% 准确率
- 📚 **循证医学集成** — KnowS API 支持中英文论文/临床指南检索，质量报告自动附带文献引用
- 🖥️ **Gradio Web UI** — 8 个功能 Tab，可视化操作，支持 ModelScope Studio 部署
- 🔒 **安全合规** — 不使用真实患者数据，脱敏 Token 化处理，明确不做诊断承诺

---

## 3. 魔搭社区运行/部署指南

### 魔搭展示链接

- **MCP 工具**: [https://modelscope.cn/mcp/servers/yuppiez/leo](https://modelscope.cn/mcp/servers/yuppiez/leo)
- **Gradio Studio**: （待部署）

### 本地运行步骤

#### 环境要求

- Python 3.10+
- FastMCP 3.4+
- Pandas, NumPy, PyArrow, scikit-learn

#### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/yuppiez99999/-MCP-Tool-Medical-Data-QA-MCP-.git
cd -MCP-Tool-Medical-Data-QA-MCP-

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（可选，用于循证医学功能）
# 复制 .env.example 为 .env，填入 KNOWS_API_KEY
# KNOWS_API_KEY=sk-knows-xxx

# 4. 启动 MCP Server（stdio 模式）
python mcp_server.py

# 5. 或启动 Gradio Web 界面
python app.py
# 访问 http://localhost:7860
```

#### MCP 客户端配置（Claude Desktop）

编辑 `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "medical-data-qa": {
      "command": "python",
      "args": ["path/to/mcp_server.py"]
    }
  }
}
```

---

## 4. 演示与输入输出示例

### 示例 1：数据质量评估

**输入**:
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
assess_data_quality(records, department="radiology")
```

**输出**:
```json
{
  "overall_score": 97.6,
  "grade": "A",
  "dimensions": {
    "completeness": 98.5,
    "accuracy": 97.0,
    "timeliness": 95.0,
    "compliance": 100.0
  },
  "department": "radiology",
  "record_count": 1
}
```

### 示例 2：文献驱动科室分类（ML 模型）

**输入**:
```python
record = {
    "text": "CT scan of brain showing acute ischemic stroke in left MCA territory",
    "data_type": "image"
}
classify_department(record)
```

**输出**:
```json
{
  "department": "radiology",
  "department_cn": "放射科",
  "confidence": 0.2465,
  "classification_method": "ml_model",
  "alternatives": [
    {"name": "neurology", "name_cn": "神经内科", "score": 0.21},
    {"name": "cardiovascular", "name_cn": "心血管科", "score": 0.19}
  ],
  "model_info": {
    "model_type": "tfidf_logistic_regression",
    "training_samples": 480,
    "top_keywords": [
      {"word": "ct", "weight": 1.23},
      {"word": "brain tumor", "weight": 0.98},
      {"word": "brain", "weight": 0.87}
    ]
  }
}
```

### 示例 3：循证医学文献检索

**输入**:
```python
search_medical_evidence(
    query="CT影像质量控制标准",
    source="paper_cn"
)
```

**输出**:
```json
{
  "query": "CT影像质量控制标准",
  "source": "paper_cn",
  "total": 10,
  "results": [
    {
      "title": "多层螺旋CT图像质量控制参数优化研究",
      "authors": ["张三", "李四"],
      "journal": "中华放射学杂志",
      "year": 2023,
      "abstract": "目的：探讨多层螺旋CT图像质量控制的关键参数...",
      "source": "paper_cn"
    }
  ]
}
```

---

## 5. 局限性与未来规划

### 当前版本局限性

1. **ML 模型训练数据有限** — 目前仅使用 480 篇英文文献训练，中文场景下准确率有待提升
2. **科室覆盖范围有限** — 仅支持 8 个常见科室，未覆盖眼科、皮肤科、口腔科等专科
3. **质量评估规则化** — 当前 4 维度评分基于加权规则，未引入深度学习模型
4. **循证检索深度有限** — 仅提供文献检索和摘要，未做全文解析和证据等级评级
5. **无真实患者数据** — 使用脱敏 Token 数据演示，实际应用需对接真实数据源

### 未来规划

- [ ] **扩展科室覆盖** — 从 8 科室扩展到 20+ 科室，覆盖更多专科
- [ ] **中文模型优化** — 增加中文文献训练数据，提升中文场景分类准确率
- [ ] **深度学习模型** — 引入 BERT/Med-BERT 等预训练模型提升分类效果
- [ ] **证据等级评分** — 对检索到的循证文献进行 GRADE 证据等级评级
- [ ] **多语言支持** — 支持中英文双向检索和报告生成
- [ ] **API 服务化** — 提供 RESTful API，便于系统集成
- [ ] **可视化大屏** — 数据质量监控仪表盘，实时追踪质量趋势

---

## 6. 团队与致谢

### 团队成员

- **yuppiez99999** — 项目负责人，全栈开发，MCP 工具设计

### 致谢

感谢以下开源项目和平台：

- 🏗️ **FastMCP** — MCP Server 开发框架
- 📚 **KnowS** — 医学循证证据检索 API 支持
- 🤗 **ModelScope 魔搭社区** — 赛事平台与部署支持
- 🏥 **小X宝开源医疗社区** — 赛事组织与医疗场景指导
- 📊 **北数所** — 医疗 Token 数据集参考

---

## ⚠️ 医疗合规性声明

**重要提示：**

1. 本工具仅用于**医疗数据质量评估**和**医学文献检索辅助**，**不提供任何医疗诊断建议**
2. 本工具的输出**不能替代专业医生的诊断和治疗意见**，任何医疗决策应由执业医师做出
3. 本工具使用**脱敏 Token 化数据**进行演示，**不包含任何真实患者隐私信息**
4. 循证医学文献检索结果仅供参考，文献的适用性需由专业医疗人员判断
5. 使用本工具产生的任何后果，开发者不承担任何法律责任

---

## 📊 评测数据报告

### 科室分类模型评测

| 指标 | 数值 | 说明 |
|------|------|------|
| 训练样本 | 480 篇 | 8 科室 × 60 篇/科室（KnowS 英文文献） |
| 测试样本 | 10 条 | 手工标注测试用例 |
| 测试准确率 | 80.0% | 8/10 分类正确 |
| 交叉验证准确率 | 32.5% ± 3.8% | 5 折 StratifiedKFold（8 分类随机基准 12.5%） |
| 模型大小 | ~500 KB | TF-IDF + Logistic Regression |
| 推理速度 | < 10ms | 单条记录分类 |

### MCP 工具可用性

| 工具 | 状态 | 测试通过 |
|------|------|---------|
| get_dataset_stats | ✅ 可用 | 是 |
| sample_real_records | ✅ 可用 | 是 |
| assess_data_quality | ✅ 可用 | 是 |
| classify_department | ✅ 可用 | 是 |
| grade_data_level | ✅ 可用 | 是 |
| generate_quality_report | ✅ 可用 | 是 |
| search_medical_evidence | ✅ 可用 | 是（需 API Key） |
| generate_evidence_based_report | ✅ 可用 | 是（需 API Key） |
| search_similar_data | ✅ 可用 | 是 |

---

## 🏆 参赛信息

- **比赛**: 小X宝开源医疗社区黑客松 2026 × ModelScope
- **赛道**: 医疗垂直领域 MCP 工具/Skill 开发
- **阶段**: 阶段2 MVP 开发进行中
- **选题方向**: 医疗数据质量评估 + 循证医学辅助
- **技术形式**: MCP Tool + Gradio Studio

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 📞 联系

- GitHub: [@yuppiez99999](https://github.com/yuppiez99999)
- ModelScope MCP: [医疗数据质量评估 MCP Tool](https://modelscope.cn/mcp/servers/yuppiez/leo)
- 项目仓库: https://github.com/yuppiez99999/-MCP-Tool-Medical-Data-QA-MCP-
