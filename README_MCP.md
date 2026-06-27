# 医疗数据质量评估 MCP Server

> 小X宝医疗黑客松 2026 参赛作品 | 基于 Model Context Protocol 的医疗数据质量评估 + 循证医学工具

## 简介

医疗数据质量评估 MCP Server 是一个基于 [Model Context Protocol](https://modelcontextprotocol.ai/) 的轻量服务（v1.2.0），为 AI Agent 提供医疗数据质量评估和循证医学检索能力。支持 8 大科室自动分类、4 维度质量评分、A/B/C/D 等级评定，并集成 KnowS 医学循证 API，支持中英文论文/临床指南/临床试验/药品说明书检索，质量报告可自动附带文献引用。

数据基础：390 万条医疗健康 Token 数据（北数所合规），覆盖 8 大科室。内置基于 480 篇 KnowS 英文文献训练的 TF-IDF + LR 科室分类模型，准确率 83.3%。

## 8 大 MCP 工具

| 工具 | 描述 | 必填参数 |
|------|------|---------|
| `assess_data_quality` | 评估医疗数据质量：4维度评分+综合分+等级 | `records` |
| `classify_department` | 自动识别医疗科室（ML模型优先，规则引擎兜底） | `record` |
| `grade_data_level` | 数据等级评定（A/B/C/D级）+推荐用途 | `quality_score` |
| `generate_quality_report` | 生成完整质量报告（科室分布+维度分析+改进建议） | `records` |
| `search_similar_data` | 检索与给定质量画像相似的历史数据 | `quality_profile` |
| `search_medical_evidence` | KnowS 医学循证检索：论文/指南/试验/药品说明书 | `query` |
| `assess_with_evidence` | 质量评估+文献检索联动：自动生成检索词并检索相关文献 | `records` |
| `generate_evidence_based_report` | 带循证文献引用的完整质量报告 | `records` |

## 快速开始

### 环境要求

- Python 3.10+
- FastMCP 3.4+
- Pandas、NumPy、PyArrow、scikit-learn
- `requests`（KnowS 循证检索需要）

### 安装

```bash
pip install fastmcp pandas numpy pyarrow scikit-learn requests python-dotenv
```

### 配置环境变量（循证检索功能需要）

```bash
# 复制 .env.example 为 .env
cp .env.example .env

# 编辑 .env，填入 KnowS API Key
KNOWS_API_KEY=sk-knows-xxx
```

### 运行 MCP Server

```bash
python mcp_server.py
```

### MCP 客户端配置

在 Claude Desktop 或其他 MCP 客户端的配置文件中添加：

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

## 工具使用示例

### 1. 评估数据质量

```python
records = [
    {
        "completeness": 95, "accuracy": 92,
        "timeliness": 88, "compliance": 96,
        "data_type": "image", "department": "radiology"
    },
    {
        "completeness": 60, "accuracy": 65,
        "timeliness": 55, "compliance": 70,
        "data_type": "lab", "department": "laboratory"
    }
]
result = server.assess_data_quality(records)
# 返回: average_quality_score, level_distribution,
#       department_distribution, details (含逐条建议)
```

### 2. 科室自动分类（ML 模型）

```python
record = {
    "text": "CT scan of brain showing acute ischemic stroke",
    "data_type": "image"
}
result = server.classify_department(record)
# 返回: primary_department, department_cn, confidence,
#       alternatives, classification_method (ml_model/rule_based),
#       model_info (model_type, training_samples, top_keywords)
```

### 3. KnowS 循证医学检索

```python
# 检索中文论文
result = server.search_medical_evidence(
    query="CT影像质量控制标准",
    source="paper_cn",
    max_results=10
)

# 检索临床指南
result = server.search_medical_evidence(
    query="diabetes management guideline",
    source="guide",
    max_results=5
)

# 支持的数据源: paper_en / paper_cn / meeting / guide / trial / package_insert
```

### 4. 质量评估 + 文献联动

```python
result = server.assess_with_evidence(
    records=records,
    source="paper_en",
    evidence_count=5
)
# 自动根据科室和质量薄弱维度生成检索词，
# 检索相关文献作为改进依据
```

### 5. 带文献引用的报告

```python
result = server.generate_evidence_based_report(
    records=records,
    dataset_name="放射科测试数据集",
    evidence_per_dimension=2,
    source="paper_en"
)
# 每个质量薄弱维度自动检索对应文献，
# 生成带编号引用的改进建议
```

## 质量评估标准

### 4 维度加权评分

| 维度 | 权重 | 说明 |
|------|------|------|
| 完整性 (completeness) | 30% | 字段缺失率、关键字段覆盖 |
| 准确性 (accuracy) | 35% | 数据交叉验证、逻辑一致性 |
| 时效性 (timeliness) | 15% | 数据更新频率、时效窗口 |
| 合规性 (compliance) | 20% | 知情同意、脱敏处理、审计溯源 |

### 数据等级

| 等级 | 分数 | 描述 | 推荐用途 | 定价系数 |
|------|------|------|---------|---------|
| A级 | 90-100 | 高质量，可直接训练 | 模型训练、临床决策支持 | 1.50x |
| B级 | 75-89 | 良好质量，清洗后可用 | 模型预训练、数据分析 | 1.00x |
| C级 | 60-74 | 基础质量，需人工审核 | 统计分析、趋势研究 | 0.60x |
| D级 | <60 | 质量不达标 | 仅限内部参考 | 0.30x |

## 科室分类模型

### ML 模型信息

| 指标 | 数值 |
|------|------|
| 模型类型 | TF-IDF + Logistic Regression |
| 训练样本 | 480 篇（8 科室 × 60 篇，KnowS 英文文献） |
| 测试样本 | 24 条手工标注 |
| 测试准确率 | 83.3%（20/24） |
| 交叉验证 | 71.67% ± 3.12%（5折） |
| 别名映射 | 4 条规则（cardiovascular→cardiology 等） |
| 模型大小 | ~500 KB |
| 推理速度 | < 10ms/条 |

### 8 大科室

| 科室键 | 中文名 | 权重 | 说明 |
|--------|--------|------|------|
| radiology | 放射科 | 1.50 | 影像数据稀缺，AI 训练需求高 |
| pathology | 病理科 | 1.40 | 病理标注数据专业度高 |
| neurology | 神经内科 | 1.35 | 脑电/神经影像价值 |
| cardiology | 心血管科 | 1.30 | ECG 数据临床价值高 |
| laboratory | 检验科 | 1.00 | 检验报告结构化 |
| orthopedics | 骨科 | 0.95 | X-Ray 影像 |
| pediatrics | 儿科 | 0.90 | 儿童发育数据 |
| emergency | 急诊科 | 0.85 | 分诊记录 |

## KnowS 循证医学集成

循证检索功能通过 KnowS API 实现，支持以下文献数据源：

| 数据源 | 键 | 最大返回 | 说明 |
|--------|-----|---------|------|
| 英文论文 | `paper_en` | 40 | 英文医学期刊论文 |
| 中文论文 | `paper_cn` | 40 | 中文医学期刊论文 |
| 会议论文 | `meeting` | 5 | 学术会议论文 |
| 临床指南 | `guide` | 5 | 临床实践指南 |
| 临床试验 | `trial` | 5 | 注册临床试验 |
| 药品说明书 | `package_insert` | 5 | 药品说明书 |

返回的每条文献包含：标题、摘要、期刊、出版日期、作者、DOI、研究类型、影响因子、中科院分区、WOS 分区、PDF 可用状态。

使用时需配置 `KNOWS_API_KEY` 环境变量。未配置时不报错，返回提示信息。

## 数据基础

| 指标 | 数值 |
|------|------|
| 数据总量 | 390 万条医疗健康 Token 记录 |
| 数据来源 | 北数所合规医疗 Token 数据集 |
| 覆盖科室 | 8 大科室 |
| 数据类型 | 8 种（影像/文本/心电/检验/病理/基因/生命体征/其他） |
| 质量分级 | A 级(83.4%) / B 级(16.6%) |
| 隐私合规 | 无真实患者信息，纯 Token 结构 |

## 性能指标

| 指标 | 数值 |
|------|------|
| 工具响应时间 | < 50ms（单次调用，不含 KnowS API） |
| KnowS 检索耗时 | 1-30s（取决于 API 响应） |
| ML 分类耗时 | < 10ms/条 |
| 内存占用 | ~50MB |

## 技术栈

- **MCP 框架**: 轻量自实现（零第三方 MCP 依赖）
- **传输协议**: stdio
- **ML 模型**: scikit-learn（TF-IDF + Logistic Regression）
- **循证检索**: KnowS API（RESTful）
- **数据处理**: Pandas + NumPy
- **文件格式**: pickle（模型）、Parquet（数据缓存）

## 医疗合规性声明

重要提示：

1. 本工具仅用于医疗数据质量评估和医学文献检索辅助，不提供任何医疗诊断建议
2. 本工具的输出不能替代专业医生的诊断和治疗意见，任何医疗决策应由执业医师做出
3. 本工具使用脱敏 Token 化数据进行演示，不包含任何真实患者隐私信息
4. 循证医学文献检索结果仅供参考，文献的适用性需由专业医疗人员判断
5. 使用本工具产生的任何后果，开发者不承担任何法律责任

## 参赛信息

- **比赛**: 小X宝开源医疗社区黑客松 2026 × ModelScope
- **赛道**: 医疗垂直领域 MCP 工具/Skill 开发
- **技术形式**: MCP Tool + Gradio Studio
- **魔搭 MCP**: [https://modelscope.cn/mcp/servers/yuppiez/leo](https://modelscope.cn/mcp/servers/yuppiez/leo)

## 许可证

MIT License — 允许商用、修改、分发

## 作者

**yuppiez99999** — 项目负责人，全栈开发，MCP 工具设计
