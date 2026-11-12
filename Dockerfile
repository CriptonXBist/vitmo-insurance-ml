FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Команда запуска будет добавлена позже
CMD ["echo", "Dockerfile готов к настройке"]
