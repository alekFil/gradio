import os

import gradio as gr
from inferences.inference_elements import predict
from inferences.inference_landmarks import LandmarksProcessor
from utils import utils as u

LANDMARK_MODELS = {
    "Lite": "app/inferences/models/landmarkers/pose_landmarker_lite.task",
    "Full": "app/inferences/models/landmarkers/pose_landmarker_full.task",
    "Heavy": "app/inferences/models/landmarkers/pose_landmarker_heavy.task",
}


def process_video(
    video_file,
    draw_mode,
    quality_mode,
    progress=gr.Progress(track_tqdm=True),
):
    # Генерируем хеш видеофайла
    video_hash = u.generate_video_hash(video_file)

    # Проверяем кэш на наличие предсказанных скелетных данных
    landmarks_data, world_landmarks_data = u.load_cached_landmarks(video_hash)
    if landmarks_data is None:
        # Если данных нет в кэше, запускаем процесс расчета и сохраняем результат
        landmarks_data, world_landmarks_data, figure_masks_data = LandmarksProcessor(
            LANDMARK_MODELS["Lite"],
            video_hash,
            do_resize=True,
        ).process_video(video_file, step=3)

        # Сохраняем landmarks_data в кэш
        u.save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data)
    else:
        print("Данные landmarks загружены из кэша.")

    print(landmarks_data[0], world_landmarks_data[0], sep="\n")

    predicted_labels, _ = predict(landmarks_data, world_landmarks_data)
    print(f"{predicted_labels[405:420]=}")

    return os.path.join("app/output/processed_video_compatible.mp4")
