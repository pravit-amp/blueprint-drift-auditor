FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py index.html start.sh ./
RUN chmod +x start.sh

EXPOSE 3000

CMD ["./start.sh"]
