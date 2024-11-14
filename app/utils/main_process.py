import re
from time import time

import gradio as gr
from inferences.inference_elements import predict
from inferences.inference_landmarks import LandmarksProcessor
from utils import utils as u
from utils.logger import setup_logger
from utils.reels_processor import ReelsProcessor

# Инициализируем логгер
logger = setup_logger("main", "app/resources/logs/main.log")

LANDMARK_MODELS = {
    "Lite": "app/inferences/models/landmarkers/pose_landmarker_lite.task",
    "Full": "app/inferences/models/landmarkers/pose_landmarker_full.task",
    "Heavy": "app/inferences/models/landmarkers/pose_landmarker_heavy.task",
}

hash_pattern = re.compile(r"/([a-f0-9]{64})/")


def process_video(
    video_file,
    draw_mode,
    quality_mode,
    progress=gr.Progress(track_tqdm=True),
):
    gradio_dirname = hash_pattern.search(video_file).group(1)
    logger.debug(f"Начата работа с видео: {gradio_dirname}")
    gr.update()
    # Генерируем хеш видеофайла
    video_hash = u.generate_video_hash(video_file)
    logger.debug(f"Сгенерирован хэш видео: {video_hash=}")
    gr.update()

    # Проверяем кэш на наличие предсказанных скелетных данных
    landmarks_data, world_landmarks_data = u.load_cached_landmarks(video_hash)
    if landmarks_data is None:
        # Если данных нет в кэше, запускаем процесс расчета и сохраняем результат
        logger.info("Данных landmarks в кэше не имеется.")
        logger.debug(f"Начало расчета landmarks: {time()}")
        gr.update()
        landmarks_data, world_landmarks_data, figure_masks_data = LandmarksProcessor(
            LANDMARK_MODELS["Lite"],
            video_hash,
            do_resize=True,
        ).process_video(video_file, step=3)
        logger.debug(f"Окончание расчета landmarks: {time()}")
        gr.update()

        # Сохраняем landmarks_data в кэш
        u.save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data)
        logger.info("Данные landmarks кэшированы.")
        gr.update()
    else:
        logger.info("Данные landmarks загружены из кэша.")
        gr.update()

    print(landmarks_data[0], world_landmarks_data[0], sep="\n")

    predicted_labels, _ = predict(landmarks_data, world_landmarks_data)
    print(f"{predicted_labels[405:420]=}")

    reels_fragments = u.find_reels_fragments(predicted_labels, 1, 25)
    print(reels_fragments)
    reels = [(x * 3, y * 3) for x, y in reels_fragments]

    reels_processor = ReelsProcessor(video_file, step=3)
    processed_video = reels_processor.process_jumps(
        tuple(reels),
        landmarks_data,
        padding=0,
        draw_mode=draw_mode,
    )

    print("Обработанное видео сохранено как:", processed_video)

    return processed_video
