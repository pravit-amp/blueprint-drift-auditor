FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py index.html ./

EXPOSE 3000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-3000}
