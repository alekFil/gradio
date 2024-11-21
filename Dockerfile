FROM python:3.12-slim-bullseye

# Установка ffmpeg
RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    libgl1 \
    libglib2.0-0 \
    && apt-get remove -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar -xf ffmpeg-release-amd64-static.tar.xz \
    && mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ \
    && mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ \
    && rm -rf ffmpeg-release-amd64-static* \
    && apt-get purge -y wget xz-utils

# Установка зависимостей python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Создание необходимых директорий
RUN mkdir -p app/resources/landmarks_cache \
    app/resources/logs \
    app/resources/outputs \
    app/resources/crops_cache \
    app/resources/preds_cache \
    app/resources/tmp
    
COPY . .

# Указываем порты, которые нужно открыть
EXPOSE 1328

# Команда для запуска Gradio
CMD ["python3", "app/main.py"]
