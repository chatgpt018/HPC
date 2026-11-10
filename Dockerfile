# Application container only. HPC MPI worker environment is intentionally separate.
FROM python:3.11-slim
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MARKETINTEL_DATA_DIR=/workspace/storage \
    MARKETINTEL_BIND=0.0.0.0 \
    MARKETINTEL_PORT=7860
COPY pyproject.toml README.md ./
COPY marketintel/ ./marketintel/
COPY app.py ./
RUN python -m pip install --no-cache-dir -e .
RUN mkdir -p /workspace/storage
EXPOSE 7860
CMD ["python", "app.py"]
