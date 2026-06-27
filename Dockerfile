# ModelScope MCP 可托管部署 Dockerfile
# 基于 Gradio 5.29.0 + MCP 协议

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖 (gradio + pandas + sklearn 编译/运行所需)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ make \
    libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 7860

# 启动 Gradio + MCP 服务
CMD ["python", "app.py"]
