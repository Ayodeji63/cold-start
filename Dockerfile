FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY data/metadata ./data/metadata

EXPOSE 8001

CMD ["uvicorn", "app.NaijaRec_ColdStart:app", "--host", "0.0.0.0", "--port", "8001"]
