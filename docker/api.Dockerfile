FROM python:3.12-slim

WORKDIR /app

# Las dependencias se instalan primero para aprovechar la cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El código del engine vive bajo multimodal-db
COPY multimodal-db/ .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
