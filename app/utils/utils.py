import hashlib
import os
import pickle

CACHE_DIR = "app/landmarks_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


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
