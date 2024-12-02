import re

from inferences.inference_elements import predict
from inferences.inference_landmarks import LandmarksProcessor
from utils import utils as u
from utils.logger import setup_logger
from utils.process_landarks import process_landmarks
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
def calculate_landmarks(video_file, video_hash, calculate_type="pre"):
    landmarks_data, world_landmarks_data, figure_masks_data, step = LandmarksProcessor(
        LANDMARK_MODELS["Lite"],
        video_hash,
        do_resize=True,
    ).process_video(video_file, calculate_type)
    return landmarks_data, world_landmarks_data, figure_masks_data, step


def process_video(
    video_file,
    draw_mode,
    quality_mode,
    progress=gr.Progress(track_tqdm=True),
):
    gradio_dirname = hash_pattern.search(video_file).group(1)
    logger.debug(f"Начата работа с видео: {gradio_dirname}")

    # Генерируем хеш видеофайла
    video_hash = u.generate_video_hash(video_file)
    logger.debug(f"Сгенерирован хэш видео: {video_hash}")

    # Проверяем кэш на наличие предсказанных скелетных данных
    landmarks_data, world_landmarks_data = u.load_cached_landmarks(video_hash)
    if landmarks_data is None:
        # Если данных нет в кэше, запускаем процесс расчета и сохраняем результат
        logger.info("Данных landmarks в кэше не имеется.")
        # landmarks_data, world_landmarks_data, _, step = calculate_landmarks(
        #     video_file,
        #     video_hash,
        #     calculate_type="pre",
        # )
        landmarks_data, world_landmarks_data, _, step = process_landmarks(
            video_file,
            video_hash,
        )

        with open(f"app/resources/landmarks_cache/{video_hash}_lms.txt", "w") as f:
            f.write(str(list(landmarks_data)))

        # Сохраняем landmarks_data в кэш
        # u.save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data)
        # logger.info("Данные landmarks кэшированы.")
    else:
        logger.info("Данные landmarks загружены из кэша.")

    predicted_labels, predicted_probs = predict(landmarks_data, world_landmarks_data)

    with open(f"app/resources/preds_cache/{video_hash}_labels.txt", "w") as f:
        f.write(str(list(predicted_labels.cpu().numpy())))

    with open(f"app/resources/preds_cache/{video_hash}_probs.txt", "w") as f:
        f.write(str(list(predicted_probs.cpu().numpy())))

    reels_fragments = u.find_reels_fragments(predicted_labels, 1, 25)
    # Найдем вращения
    spins_fragments = u.find_reels_fragments(predicted_labels, 2, 25)
    if len(reels_fragments) != 0:
        logger.info(f"Обнаружено {len(reels_fragments)} прыжка(-ов) (указаны кадры):")
    else:
        logger.info("Не обнаружено прыжков. Работа завершена")
        return "Error"
    print(reels_fragments)
    print(spins_fragments)
    reels = [(x * step, y * step) for x, y in reels_fragments]
    logger.info(f"{reels}")

    reels_processor = ReelsProcessor(video_file, video_hash, step=step)
    # processed_video = reels_processor.process_jumps(
    #     tuple(reels),
    #     landmarks_data,
    #     padding=0,
    #     draw_mode=draw_mode,
    #     video_hash=video_hash,
    # )
    processed_video = reels_processor.process_elements_multiprocessing(
        tuple(reels),
        landmarks_data,
        video_hash=video_hash,
    )

    print("Обработанное видео сохранено как:", processed_video)

    return (processed_video, processed_video, processed_video)
