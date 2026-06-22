# 数据资产登记报告 · 医疗健康 A 级 / B 级 Token 数据集

- **登记 ID**: BJDE-HEALTH-20260622143312
- **登记机构**: 北京国际大数据交易所（北数所）
- **数据提供方**: 医疗数据联盟
- **数据格式**: CSV / JSON
- **数据集规模**: 5,000,000 条 Token
- **登记时间**: 2026-06-22 14:33:12

## 一、合规性声明

本数据集严格遵循《数据安全法》《个人信息保护法》《数据出境安全评估办法》等法律法规。
数据已完成去标识化处理，token_id 与实体信息经过单向哈希处理，合规评分均达 95.0 分以上。
可用于医疗 AI 模型训练、数据资产入表、医学研究等合规场景。

## 二、数据质量指标

| 指标 | 值 |
|------|-----|
| 质量分范围 | 94.00 - 100.00 |
| 质量分均值 | 97.00 |
| 等级 A 占比 | 4169292 (83.39%) |
| 等级 B 占比 | 830708 (16.61%) |

### 科室 Token 分布

| 科室 (category) | 中文名 | Token 数 | 占比 |
|------------------|-------|---------|------|
| radiology | 放射科 | 625000 | 12.50% |
| laboratory | 检验科 | 625000 | 12.50% |
| pathology | 病理科 | 625000 | 12.50% |
| cardiology | 心血管科 | 625000 | 12.50% |
| neurology | 神经内科 | 625000 | 12.50% |
| orthopedics | 骨科 | 625000 | 12.50% |
| pediatrics | 儿科 | 625000 | 12.50% |
| emergency | 急诊科 | 625000 | 12.50% |

## 三、数据资产价值估算

- 合格 Token 数：100,000
- 总价值（单条参考价）：¥ 4,900,275.92
- 总价值（企业批量折扣价）：¥ 4,165,219.43
- 平均每条价值：¥ 49.00

### 按科室价值分布

| 科室 | category | count | total_value | avg_value |
|------|----------|-------|-------------|-----------|
| 放射科 | radiology | 100000 | ¥ 4,900,275.92 | ¥ 49.00 |

## 四、数据字段字典

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| token_id | string | Token 唯一标识，用于合规审计与溯源 | `HEALTH-XXXX-XXXX` |
| domain | string | 数据所属领域 | `healthcare` |
| category | string | 医疗科室类别（放射科/检验科/病理科/心血管科/神经内科/骨科/儿科/急诊科） | `radiology` |
| data_type | string | 数据形态（影像/文本/心电/检验/病理/基因/生命体征等） | `image` |
| entity_id | string | 脱敏后的实体标识，支持同一实体跨科室聚合 | `ENT-XXXX` |
| data_quality_score | float | 综合数据质量分（0-100） | `98.5` |
| token_level | string | Token 等级：A 高质量 / B 标准质量 | `A` |
| completeness | float | 数据完整性分（0-100） | `99.0` |
| accuracy | float | 数据准确性分（0-100） | `97.0` |
| timeliness | float | 时效性分（0-100） | `95.0` |
| compliance_score | float | 合规性分（0-100），≥95 视为资产合格 | `100.0` |
| created_at | datetime | Token 创建时间 | `2026-06-16 08:00:00` |

## 五、数据样本展示

| token_id | category | data_type | token_level | data_quality_score | completeness | accuracy | timeliness | compliance_score | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CBCD5D853AE1CD70 | radiology | other | A | 99.38 | 99.68 | 98.95 | 99.59 | 100.0 | 2026-06-20T15:41:09 |
| FBEEC10FAA2E95DD | radiology | other | A | 96.56 | 98.23 | 97.04 | 96.56 | 100.0 | 2026-06-20T15:41:09 |
| 02544BBA5BE0DED4 | radiology | other | A | 99.86 | 97.83 | 95.74 | 95.83 | 100.0 | 2026-06-20T15:41:09 |
| AF34448B39B6D20B | radiology | other | A | 98.64 | 97.51 | 99.19 | 98.31 | 100.0 | 2026-06-20T15:41:09 |
| 8EBE9C0ADD3EC93F | radiology | other | A | 97.89 | 99.37 | 95.45 | 95.39 | 100.0 | 2026-06-20T15:41:09 |

---
*本报告由 DataExchangeRegistrar 于 2026-06-22 14:33:12 自动生成*