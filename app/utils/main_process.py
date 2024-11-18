import re

from inferences.inference_elements import predict
from inferences.inference_landmarks import LandmarksProcessor
from utils import utils as u
from utils.logger import setup_logger
from utils.reels_processor import ReelsProcessor
from utils.utils import log_execution_time

import gradio as gr

# Инициализируем логгер
logger = setup_logger("main", "app/resources/logs/main.log")

LANDMARK_MODELS = {
    "Lite": "app/inferences/models/landmarkers/pose_landmarker_lite.task",
    "Full": "app/inferences/models/landmarkers/pose_landmarker_full.task",
    "Heavy": "app/inferences/models/landmarkers/pose_landmarker_heavy.task",
}

hash_pattern = re.compile(r"/([a-f0-9]{64})/")


@log_execution_time(logger, "Начало расчета landmarks", "Окончание расчета landmarks")
def calculate_landmarks(video_file, video_hash):
    landmarks_data, world_landmarks_data, figure_masks_data = LandmarksProcessor(
        LANDMARK_MODELS["Lite"],
        video_hash,
        do_resize=True,
    ).process_video(video_file, step=3)
    return landmarks_data, world_landmarks_data, figure_masks_data


def process_video(
    video_file,
    draw_mode,
    quality_mode,
    progress=gr.Progress(track_tqdm=True),
):
    gradio_dirname = hash_pattern.search(video_file).group(1)
    logger.debug(f"Начата работа с видео: {gradio_dirname}")

    if not u.check_video(video_file):
        logger.debug("Обнаружено нестандартное видео. Производится перекодировка")
        video_file = u.update_video(video_file)

    # Генерируем хеш видеофайла
    video_hash = u.generate_video_hash(video_file)
    logger.debug(f"Сгенерирован хэш видео: {video_hash}")

    # Проверяем кэш на наличие предсказанных скелетных данных
    landmarks_data, world_landmarks_data = u.load_cached_landmarks(video_hash)
    if landmarks_data is None:
        # Если данных нет в кэше, запускаем процесс расчета и сохраняем результат
        logger.info("Данных landmarks в кэше не имеется.")
        landmarks_data, world_landmarks_data, _ = calculate_landmarks(
            video_file,
            video_hash,
        )

        # Сохраняем landmarks_data в кэш
        u.save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data)
        logger.info("Данные landmarks кэшированы.")
    else:
        logger.info("Данные landmarks загружены из кэша.")

    predicted_labels, _ = predict(landmarks_data, world_landmarks_data)

    reels_fragments = u.find_reels_fragments(predicted_labels, 1, 25)
    if len(reels_fragments) != 0:
        logger.info(f"Обнаружено {len(reels_fragments)} прыжка(-ов) (указаны кадры):")
    else:
        logger.info("Не обнаружено прыжков. Работа завершена")
        return "Error"
    print(reels_fragments)
    reels = [(x * 3, y * 3) for x, y in reels_fragments]
    logger.info(f"{reels}")

    reels_processor = ReelsProcessor(video_file, step=3)
    processed_video = reels_processor.process_jumps(
        tuple(reels),
        landmarks_data,
        padding=0,
        draw_mode=draw_mode,
    )

    print("Обработанное видео сохранено как:", processed_video)

    return processed_video
