FROM python:3.11-slim
WORKDIR /app
COPY web_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FLASK_APP=web_service/run.py
ENV PYTHONPATH=/app/web_service
ENV PYTHONUNBUFFERED=1
EXPOSE 5001
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
