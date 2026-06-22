# 医疗数据质量评估 MCP Tool

> **小X宝医疗黑客松 2026 参赛作品** — 基于 Model Context Protocol 的医疗数据质量评估工具

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-3.4+-green.svg)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

---

## 📖 项目简介

**医疗数据质量评估 MCP Tool** 是一个基于 Model Context Protocol (MCP) 标准的医疗数据质量评估服务，为医疗 AI 模型训练、数据资产入表、合规审计等场景提供端到端的数据质量评估能力。

### 核心价值

- 🔍 **真实数据接入**: 5,000,000 条北数所合规医疗 Token 数据
- ⚡ **毫秒级响应**: 50,000 条分层采样 + Parquet 缓存
- 🎯 **7 大 MCP 工具**: 覆盖数据质量评估全流程
- 🏥 **8 大科室分类**: 放射科/病理科/神经内科/心血管科/检验科/骨科/儿科/急诊科
- 📊 **4 维度评分**: 完整性 + 准确性 + 时效性 + 合规性

---

## 🛠️ MCP 工具列表

| 工具名称 | 功能描述 | 输入 | 输出 |
|---------|---------|------|------|
| `get_dataset_stats` | 获取真实数据集统计信息 | 无 | 数据集规模/等级分布/科室分布 |
| `sample_real_records` | 采样真实数据记录 | department, n | 真实 Token 记录列表 |
| `assess_data_quality` | 评估医疗数据质量 | records, department | 4 维度评分 + A/B/C/D 等级 |
| `classify_department` | 自动科室分类 | record | 科室 + 置信度 |
| `grade_data_level` | 数据等级评定 | quality_score | A/B/C/D 等级 |
| `generate_quality_report` | 生成完整质量报告 | records | Markdown 报告 |
| `search_similar_data` | 检索相似数据 | record, top_k | 相似记录 + 相似度 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- FastMCP 3.4+
- Pandas, NumPy, PyArrow

### 安装

```bash
git clone https://github.com/yuppiez99999/-MCP-Tool-Medical-Data-QA-MCP-.git
cd -MCP-Tool-Medical-Data-QA-MCP-
pip install -r requirements.txt
```

### 启动 MCP Server

```bash
python server.py
```

Server 将通过 stdio 协议启动，等待 MCP 客户端连接。

---

## 🔗 MCP 客户端配置

### Claude Desktop

编辑 `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "medical-data-qa": {
      "command": "python",
      "args": ["e:/各种PY程序/18-医疗AI模型系统/server.py"]
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
      "args": ["./server.py"],
      "cwd": "./18-医疗AI模型系统"
    }
  }
}
```

### ModelScope MCP 广场

已在 ModelScope MCP 广场上线，可直接订阅使用。

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

### 3. 检索相似数据

```python
record = {"category": "radiology", "data_type": "ct_image", "data_quality_score": 98.5}
result = search_similar_data(record, top_k=5)
# 返回: 5条最相似的真实Token记录 + 相似度分数
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

## 🧪 测试

```bash
python test_real_data.py
```

**测试结果**: 7/7 全部通过 ✅

```
[测试1] get_dataset_stats      ✅ 5,000,000条 / 50,000采样
[测试2] sample_real_records    ✅ 真实token_id: 68D3B03359CA4B7F
[测试3] search_similar_data    ✅ 相似度0.999
[测试4] assess_data_quality    ✅ 3条记录评估正确
[测试5] classify_department    ✅ 置信度0.98
[测试6] grade_data_level       ✅ A/B/C/D等级正确
[测试7] generate_quality_report ✅ 完整报告生成
```

---

## 📁 项目结构

```
18-医疗AI模型系统/
├── server.py                    # MCP Server 入口 (FastMCP + stdio)
├── mcp_server.py                # MCP 核心逻辑
├── app.py                       # Gradio Web 界面
├── modelscope.json              # ModelScope 部署配置
├── config.yaml                  # 全局配置
├── requirements.txt             # Python 依赖
├── test_real_data.py            # 集成测试
├── data/
│   └── loader.py                # 真实数据加载器 (分层采样+Parquet)
├── models/
│   └── classifier.py            # 多任务学习分类器
├── modules/
│   ├── audit_trail.py           # 合规审计
│   ├── data_exchange.py         # 北数所登记
│   └── ...
├── api/
│   └── healthcare_ai_extension.py
└── outputs/                     # 运行时生成 (git忽略)
    ├── data_sample.parquet      # 50,000条采样缓存
    └── data_stats.json           # 统计信息
```

---

## 🎯 应用场景

1. **医疗 AI 模型训练** — 筛选高质量 Token 用于模型训练
2. **数据资产入表** — 为医疗机构数据资产估值提供定价基准
3. **合规审计溯源** — 通过 token_id 实现全链路追踪
4. **多模态医学研究** — 跨科室 Token 聚合分析
5. **北数所数据产品登记** — 符合数据资产登记规范

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
- ModelScope: [医疗数据质量评估 MCP Tool](https://modelscope.cn/mcp/servers/create)
