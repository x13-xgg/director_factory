# 全自动导演工厂 — 生产级 Docker 镜像
# 多阶段构建: builder → runtime

FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --user -e . 2>/dev/null || true
RUN pip install --no-cache-dir --user \
    openai anthropic httpx python-dotenv pydantic rich pillow numpy soundfile \
    asyncpg redis fakeredis uvicorn starlette prometheus-client

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="director-factory"
LABEL org.opencontainers.image.description="全自动导演工厂 — 多 Agent 协作视频生产系统"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV DIRECTOR_MODE=production

RUN mkdir -p /app/outputs /app/assets /app/outputs/checkpoints

EXPOSE 9090

# 默认入口: API 服务模式 (可改为 CLI)
ENTRYPOINT ["python", "-m", "src.main"]
