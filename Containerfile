FROM python:3.11-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# chat session persistence (v1.7.0): SQLite DB lives at /data (PVC-mounted)
RUN mkdir -p /data
COPY app.py .
COPY ui.html .
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
