# texturr service: REST + MCP, CPU only. The embedding model is baked in at build time so the
# running container never needs the network (it runs with HF_HUB_OFFLINE=1).
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/opt/hf

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt /tmp/
RUN pip install -r /tmp/requirements-service.txt

ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')" \
 && chmod -R a+rX /opt/hf

WORKDIR /app
COPY texturr.py llm.py models.py clustering.py clean.py report.py session.py server.py ./

RUN useradd --system --uid 10001 --no-create-home texturr
USER texturr
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=3)"

# Binding all interfaces is intended inside the private compose network; the port is not published.
CMD ["python", "texturr.py", "serve", "--host", "0.0.0.0", "--allow-network", "--offline"]
