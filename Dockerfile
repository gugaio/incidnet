# syntax=docker/dockerfile:1

FROM python:3.12-slim

# - PYTHONUNBUFFERED: logs vão direto para o stdout (bom para docker logs)
# - PYTHONDONTWRITEBYTECODE: não gera .pyc dentro do container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instala as dependências primeiro (melhor cache de camadas). O README é
# necessário porque o pyproject.toml o referencia em `readme = "README.md"`.
COPY pyproject.toml README.md ./
COPY app ./app
COPY templates ./templates

# Instala como editable para que PROJECT_ROOT em app/config.py resolva
# para /app (não para site-packages), mantendo templates/ acessível.
RUN pip install -e .

# Usuário não-root com o diretório de dados já com dono correto. O volume
# montado em /app/workspace herda essa propriedade na primeira criação.
RUN useradd --create-home --uid 1000 incidnet \
    && mkdir -p /app/workspace \
    && chown -R incidnet:incidnet /app

USER incidnet

EXPOSE 8000

# Verifica a saúde da aplicação usando a rota /health (sem depender de curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
