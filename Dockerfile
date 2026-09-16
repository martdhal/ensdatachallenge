FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser aml ./aml
COPY --chown=appuser:appuser demo ./demo
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser streamlit_app.py .
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
