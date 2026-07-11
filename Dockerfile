FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && /usr/local/bin/python -c "import fastapi, uvicorn, openai"

COPY main.py index.html ./

EXPOSE 3000

CMD ["/usr/local/bin/python", "/app/main.py"]
