FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install uv
COPY . .

RUN uv sync
EXPOSE 8501

ENV PATH="/app/.venv/bin:${PATH}"

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py"]