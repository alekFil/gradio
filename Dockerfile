FROM python:3.12-slim-bullseye

# Установка ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Установка зависимостей python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Создание необходимых директорий
RUN mkdir -p app/resources/landmarks_cache \
    app/resources/logs \
    app/resources/outputs \
    app/resources/preds_cache \
    app/resources/tmp
    
COPY . .

# Указываем порты, которые нужно открыть
EXPOSE 1328

# Команда для запуска Gradio
CMD ["python3", "app/main.py"]
