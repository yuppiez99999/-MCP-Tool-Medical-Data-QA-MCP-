# 医疗数据质量评估 MCP Server

> 小X宝医疗黑客松 2026 参赛作品 | 基于 **5,000,000 条** 真实医疗Token数据的智能质量评估MCP工具

## 简介

医疗数据质量评估 MCP Server 是一个基于 [Model Context Protocol](https://modelcontextprotocol.ai/) 的服务，为AI Agent提供医疗数据质量评估能力。支持8大科室自动分类、4维度质量评分、A/B/C/D等级评定、完整质量报告生成，**接入真实500万条北数所合规数据集**。

## 功能特点

- **真实数据集** — 接入 5,000,000 条 A级/B级 医疗Token数据（北数所合规数据）
- **高效采样** — 50,000条分层采样 + Parquet缓存，毫秒级响应
- **质量评估** — 评估医疗数据完整性、准确性、时效性、合规性
- **科室分类** — 自动识别8大医疗科室（放射科/病理科/神经内科/心血管科/检验科/骨科/儿科/急诊科）
- **等级评定** — A级(高质量)/B级(良好)/C级(基础)/D级(不达标)
- **质量报告** — 生成含科室分布、维度分析、改进建议的完整报告
- **相似检索** — 基于真实Token_id检索相似质量画像的历史数据
- **数据采样** — 从真实数据集中随机采样记录用于预览和测试

## 数据基础

| 指标 | 数值 |
|------|------|
| 数据集总行数 | 5,000,000 条 |
| 数据来源 | 北数所A级/B级医疗Token数据集 |
| 采样规模 | 50,000 条（分层采样） |
| A级数据 | 4,169,292 条 (83.4%) |
| B级数据 | 830,708 条 (16.6%) |
| 覆盖科室 | 8大医疗科室 |
| 数据类型 | 8种（CT影像/血液检验/病理切片/心电图/超声/X光/生长记录/分诊） |
| 质量分范围 | 94.00 - 100.00 |
| 合规分 | 100.00 (全部合规) |

### 数据字段

| 字段 | 类型 | 说明 |
|------|------|------|
| token_id | string | Token唯一标识 |
| domain | string | 数据领域 (healthcare) |
| category | string | 医疗科室 |
| data_type | string | 数据类型 |
| entity_id | string | 实体标识（脱敏） |
| data_quality_score | float | 综合质量分 (94-100) |
| token_level | string | 等级 (A/B) |
| completeness | float | 完整性分 |
| accuracy | float | 准确性分 |
| timeliness | float | 时效性分 |
| compliance_score | float | 合规性分 |
| created_at | datetime | 创建时间 |

## 快速开始

### 安装

```bash
pip install fastmcp pandas numpy pyarrow
```

### 首次启动（生成采样缓存）

```bash
# 从500万条CSV中分层采样5万条，缓存为Parquet
python data/loader.py
```

### 运行 MCP Server

```bash
python server.py
```

### MCP 客户端配置

在 MCP 客户端（如 Claude Desktop、Cursor、Cherry Studio）的配置文件中添加：

```json
{
  "mcpServers": {
    "medical-data-qa": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/medical-data-qa-mcp"
    }
  }
}
```

## 工具列表（7个）

| 工具 | 描述 | 参数 | 数据源 |
|------|------|------|--------|
| `get_dataset_stats` | 获取数据集统计信息 | 无 | 真实数据 |
| `assess_data_quality` | 评估医疗数据质量 | records, department? | 输入数据 |
| `classify_department` | 自动科室分类 | record | 输入数据 |
| `grade_data_level` | 数据等级评定 | quality_score | 算法 |
| `generate_quality_report` | 生成完整报告 | records, dataset_name? | 输入数据 |
| `search_similar_data` | 检索相似数据 | quality_profile, department?, top_k? | **真实数据** |
| `sample_real_records` | 采样真实记录 | n, department?, level? | **真实数据** |

## 使用示例

### 获取数据集统计

```python
result = await client.call_tool("get_dataset_stats", {})
# 返回: {
#   "dataset_info": {"total_rows": 5000000, "sample_size": 50000, ...},
#   "level_distribution": {"A": 41689, "B": 8311},
#   "by_department": {"radiology": {...}, "pathology": {...}, ...}
# }
```

### 采样真实记录

```python
result = await client.call_tool("sample_real_records", {
    "n": 5,
    "department": "radiology",
    "level": "A"
})
# 返回真实Token记录，包含真实token_id
```

### 检索相似数据

```python
result = await client.call_tool("search_similar_data", {
    "quality_profile": {"completeness": 98, "accuracy": 97, "timeliness": 96, "compliance": 100},
    "department": "cardiology",
    "top_k": 10
})
# 返回真实数据集中相似度最高的10条记录
```

### 评估数据质量

```python
result = await client.call_tool("assess_data_quality", {
    "records": [
        {"completeness": 98, "accuracy": 97, "timeliness": 96, "compliance_score": 100, "data_type": "ct_image"},
        {"completeness": 60, "accuracy": 65, "timeliness": 55, "compliance_score": 70, "data_type": "blood_test"},
    ]
})
```

## 质量评估标准

### 4维度加权评分

| 维度 | 权重 | 说明 |
|------|------|------|
| 完整性 | 30% | 字段缺失率、关键字段覆盖 |
| 准确性 | 35% | 数据交叉验证、逻辑一致性 |
| 时效性 | 15% | 数据更新频率、时效窗口 |
| 合规性 | 20% | 知情同意、脱敏处理、审计溯源 |

### 数据等级

| 等级 | 分数 | 描述 | 推荐用途 | 定价系数 |
|------|------|------|---------|---------|
| A级 | 90-100 | 高质量，可直接训练 | 模型训练、临床决策支持 | 1.50x |
| B级 | 75-89 | 良好质量，清洗后可用 | 模型预训练、数据分析 | 1.00x |
| C级 | 60-74 | 基础质量，需人工审核 | 统计分析、趋势研究 | 0.60x |
| D级 | <60 | 不达标 | 仅限内部参考 | 0.30x |

### 真实数据类型映射

| 数据类型 | 中文名 | 对应科室 |
|---------|--------|---------|
| ct_image | CT影像 | 放射科 |
| blood_test | 血液检验 | 检验科 |
| pathology_slide | 病理切片 | 病理科 |
| ecg | 心电图 | 心血管科 |
| ultrasound | 超声 | 放射科 |
| x_ray | X光 | 骨科 |
| growth_record | 生长记录 | 儿科 |
| triage | 分诊记录 | 急诊科 |

## 技术栈

- **MCP框架**: FastMCP 3.4+ (Python)
- **传输协议**: stdio
- **数据集**: 5,000,000条真实医疗Token数据（北数所合规数据）
- **数据缓存**: Parquet (Snappy压缩)
- **数据处理**: Pandas + NumPy
- **开源协议**: MIT

## 性能指标

| 指标 | 数值 |
|------|------|
| 首次采样耗时 | ~30秒 (500万条CSV) |
| 后续加载耗时 | <100ms (Parquet缓存) |
| 工具响应时间 | <50ms (单次调用) |
| 内存占用 | ~50MB (5万条采样) |
| 缓存文件大小 | ~2MB (Parquet) |

## 许可证

MIT License — 允许商用、修改、分发

## 贡献

**作者**: yuppiez99999

小X宝医疗黑客松 2026 参赛作品
