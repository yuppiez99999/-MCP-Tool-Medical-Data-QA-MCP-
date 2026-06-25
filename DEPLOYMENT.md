# ModelScope Studio 部署指南

## 部署方式一：通过 Git 仓库同步（推荐）

### 步骤

1. **创建 Studio**
   - 访问: https://modelscope.cn/studios/create
   - 选择 "Gradio" SDK
   - 填写项目名称：医疗数据质量评估 MCP Tool
   - 选择公开可见

2. **关联 Git 仓库**
   - 在 Studio 设置中关联 GitHub 仓库
   - 仓库地址: https://github.com/yuppiez99999/-MCP-Tool-Medical-Data-QA-MCP-.git
   - 设置自动同步

3. **配置环境变量（可选）**
   - 在 Studio 设置中添加 `KNOWS_API_KEY` 环境变量
   - 用于启用循证医学文献检索功能

4. **启动应用**
   - 等待自动部署完成
   - 访问生成的公开 URL

## 部署方式二：手动上传文件

### 需要上传的文件

```
app.py                    # Gradio 主程序
mcp_server.py             # MCP 核心逻辑
requirements_space.txt    # Python 依赖（会自动重命名为 requirements.txt）
modelscope.json           # 元数据配置
README.md                 # 说明文档
LICENSE                   # 开源协议
scripts/                  # 训练和评测脚本
data/                     # 数据加载器
models/                   # 模型相关
modules/                  # 功能模块
```

### 注意事项

- ModelScope 会自动读取根目录的 `requirements.txt` 安装依赖
- 入口文件默认为 `app.py`
- 应用监听 `0.0.0.0:7860`

## 环境变量

| 变量名 | 必须 | 说明 |
|--------|------|------|
| KNOWS_API_KEY | 否 | KnowS 循证医学 API Key，启用文献检索功能 |

## 验证部署

部署完成后，检查以下功能：

- [ ] 首页正常加载，显示8个 Tab
- [ ] 数据质量评估功能可用
- [ ] 科室分类功能可用（ML模型+规则双引擎）
- [ ] 完整报告生成功能可用
- [ ] 医学循证检索功能可用（如配置了 API Key）
- [ ] 循证质量报告功能可用（如配置了 API Key）
- [ ] MCP 工具列表页面正常展示

## 故障排查

### 依赖安装失败
- 检查 `requirements.txt` 中的包名和版本
- 尝试降低版本号

### 应用启动失败
- 查看 Studio 日志
- 确认 `app.py` 中 `launch()` 配置为 `server_name="0.0.0.0"`

### 循证检索功能不可用
- 确认 `KNOWS_API_KEY` 环境变量已正确配置
- 检查 API Key 是否有效
