import hashlib
import os
import pickle
import subprocess
from functools import wraps
from time import time

CACHE_DIR = "app/resources/landmarks_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# Декоратор для логирования времени выполнения
def log_execution_time(
    logger,
    start_msg="Начало выполнения",
    end_msg="Окончание выполнения",
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            logger.debug(f"{start_msg}: {start_time}")

            result = func(*args, **kwargs)

            end_time = time()
            elapsed_time = end_time - start_time
            logger.debug(f"{end_msg}: {end_time}")
            logger.debug(f"Время выполнения {func.__name__}: {elapsed_time:.4f} секунд")

            return result

        return wrapper

    return decorator


def generate_video_hash(video_path):
    """Генерирует хеш на основе содержимого видеофайла."""
    hash_md5 = hashlib.md5()

    # Включаем содержимое файла в хэш
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_cache_path(video_hash, data_type):
    """Возвращает путь к файлу кэша для заданного типа данных."""
    return os.path.join(CACHE_DIR, f"{video_hash}_{data_type}.pkl")


def load_cached_landmarks(video_hash):
    """Загружает landmarks_data и world_landmarks_data из кэша, если они существуют."""
    landmarks_data = world_landmarks_data = None

    landmarks_path = get_cache_path(video_hash, "landmarks")
    world_landmarks_path = get_cache_path(video_hash, "world_landmarks")

    if os.path.exists(landmarks_path):
        with open(landmarks_path, "rb") as f:
            landmarks_data = pickle.load(f)

    if os.path.exists(world_landmarks_path):
        with open(world_landmarks_path, "rb") as f:
            world_landmarks_data = pickle.load(f)

    return landmarks_data, world_landmarks_data


def save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data):
    """Сохраняет landmarks_data и world_landmarks_data в кэш."""
    landmarks_path = get_cache_path(video_hash, "landmarks")
    world_landmarks_path = get_cache_path(video_hash, "world_landmarks")

    with open(landmarks_path, "wb") as f:
        pickle.dump(landmarks_data, f)

    with open(world_landmarks_path, "wb") as f:
        pickle.dump(world_landmarks_data, f)


def find_reels_fragments(labels, target_class, batch_size):
    fragments = []

    # Параметры для поиска последовательностей
    start = None
    count = 0

    for i, label in enumerate(labels):
        if label == target_class:
            if start is None:
                start = i
            count += 1
        else:
            if start is not None and count >= 1:
                # Определяем индекс среднего элемента
                middle_index = start + count // 2

                # Определяем, к какому батчу относится средний элемент
                batch_index = middle_index // batch_size

                # Определяем начало и конец соседних батчей
                start_batch = max(0, (batch_index - 1) * batch_size)
                end_batch = min(len(labels) - 1, (batch_index + 2) * batch_size - 1)

                # Объединяем с предыдущим фрагментом, если они пересекаются
                if fragments and start_batch <= fragments[-1][1]:
                    # Обновляем конец последнего фрагмента
                    fragments[-1] = (fragments[-1][0], max(fragments[-1][1], end_batch))
                else:
                    # Добавляем новый фрагмент
                    fragments.append((start_batch, end_batch))

            # Сброс параметров
            start = None
            count = 0

    # Проверка для последней последовательности
    if start is not None and count >= 3:
        middle_index = start + count // 2
        batch_index = middle_index // batch_size
        start_batch = max(0, (batch_index - 1) * batch_size)
        end_batch = min(len(labels) - 1, (batch_index + 2) * batch_size - 1)

        # Объединяем с предыдущим фрагментом, если они пересекаются
        if fragments and start_batch <= fragments[-1][1]:
            fragments[-1] = (fragments[-1][0], max(fragments[-1][1], end_batch))
        else:
            fragments.append((start_batch, end_batch))

    return fragments


def update_video(video_file):
    """Приводит видео (перекодирует с использованием ffmpeg к горизонтальному
    соотношению 16:9, добавляя черный холст).
    Сохраняет новый файл и возвращает путь к нему"""
    # Проверка, существует ли входной файл
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Видео файл '{video_file}' не найден.")

    # Получение информации о видео
    output_file = os.path.splitext(video_file)[0] + "_16_9_2.mp4"

    # Выполнение команды FFmpeg
    try:
        command = [
            "ffmpeg",
            "-i",
            video_file,
            "-vf",
            "pad=1920:1280:(1920-iw)/2:0",
            output_file,
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ошибка при обработке видео с FFmpeg: {e}")

    # Проверка существования выходного файла
    if not os.path.exists(output_file):
        raise RuntimeError(f"Не удалось сохранить обработанное видео: {output_file}")

    return output_file
