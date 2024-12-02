import multiprocessing
import os
import subprocess
import time

import cv2
import numpy as np
from inferences.inference_landmarks import LandmarksProcessor
from utils.logger import setup_logger

# Инициализируем логгер
logger = setup_logger("main", "app/resources/logs/main.log")


def clear_output_dir(output_dir):
    """
    Удаляет все файлы в указанной директории.
    """
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def split_video(video_path, segments_num, output_dir):
    """
    Разделение видео на равные фрагменты по кадрам.

    :param video_path: Путь к исходному видео.
    :param segments_num: Количество сегментов.
    :param output_dir: Директория для сохранения сегментов.
    :return: Список путей к сегментам и информация о сегментах.
    """
    # Узнаем общее количество кадров в видео
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    total_frames = int(result.stdout.strip())
    print(f"Общее количество кадров: {total_frames}")

    # Вычисляем количество кадров на сегмент
    frames_per_segment = int(np.ceil(total_frames / segments_num))
    # frames_per_segment = 550
    print(f"Кадров на сегмент: {frames_per_segment}")

    segment_paths = []
    segment_info = []  # Список для информации о сегментах
    current_start_frame = 0
    temp_video_path = video_path

    for segment_index in range(segments_num):
        segment_path_template = os.path.join(
            output_dir, f"segment_{segment_index:03d}_%03d.mp4"
        )

        # Выполнение команды ffmpeg для разбиения текущего видео
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_video_path,
                "-f",
                "segment",
                "-segment_frames",
                str(frames_per_segment),
                "-reset_timestamps",
                "1",
                "-an",
                "-c:v",
                "copy",
                segment_path_template,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Узнаем текущее количество кадров в нулевом фрагменте видео
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "csv=p=0",
                os.path.join(output_dir, f"segment_{segment_index:03d}_000.mp4"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        frames = int(result.stdout.strip())
        print(f"Текущее количество кадров: {frames}")

        # Добавляем информацию о сегменте
        segment_info.append(
            {
                "segment": segment_index,
                "start_frame": current_start_frame,
                "end_frame": current_start_frame + frames - 1,
                "total_frames": frames,
            }
        )

        # Собираем пути к созданным сегментам
        for file_name in sorted(os.listdir(output_dir)):
            if file_name.startswith(
                f"segment_{segment_index:03d}_"
            ) and file_name.endswith(".mp4"):
                segment_paths.append(os.path.join(output_dir, file_name))

        # Обновляем путь на последний созданный сегмент для следующего запуска
        last_segment_path = segment_paths[-1]
        temp_video_path = last_segment_path

        current_start_frame += frames
        if current_start_frame > total_frames - 1:
            print("Превышено количество кадров. Завершаем деление")
            break

    # Вывод информации о сегментах
    print("\nИнформация о созданных сегментах:")
    for info in segment_info:
        print(
            f"Сегмент: {info['segment']}, "
            f"Начальный кадр: {info['start_frame']}, "
            f"Конечный кадр: {info['end_frame']}, "
            f"Всего кадров: {info['total_frames']}"
        )

    print(f"\nПути к созданным сегментах: {segment_paths}")

    filtered_segment_paths = [
        path for path in segment_paths if not path.endswith("_001.mp4")
    ]

    print(f"\nПути к расчетным сегментах: {filtered_segment_paths}")

    return filtered_segment_paths


def process_video_segment(segment_path, step, segment_start, output_dir):
    """
    Обработка отдельного сегмента видео с учётом смещения времени.

    :param segment_path: Путь к видео сегменту.
    :param step: Шаг для выборки кадров.
    :param segment_start: Смещение времени сегмента в миллисекундах.
    """
    print(f"{segment_path=}")
    segment_id = os.path.basename(segment_path).split("_")[1].split(".")[0]
    cap = cv2.VideoCapture(segment_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {segment_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{segment_id=}, {fps=}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{segment_id=}, Всего кадров в сегменте: {total_frames}")
    print(
        f"{segment_id=}, Диапазон реальных кадров в сегменте: от {(int(segment_id))*int(total_frames)} до {(int(segment_id)+1)*int(total_frames) - 1}"
    )
    start_frame = (int(segment_id)) * int(total_frames)

    processor = LandmarksProcessor(
        model_path="app/inferences/models/landmarkers/pose_landmarker_lite.task",
        key=segment_path,  # Уникальный ключ для каждого сегмента
        calculate_masks=False,
        do_resize=False,
    )

    frame_idx = 0
    timestamps = []  # Список меток времени
    check_timestamps = []  # Список меток времени
    frame_indices = []  # Список абсолютных индексов кадров
    processed_frame_indices = []

    processed_frames = 0
    print(
        f"{segment_id=}, Шаг обработки: {step}, Обрабатывается каждый {1/step:.2f} сек."
    )

    while frame_idx < total_frames:
        absolute_index = frame_idx + int(segment_start * fps / 1000)
        absolute_index = frame_idx + start_frame
        _, frame = cap.read()
        if frame is None:
            print(f"Кадр {frame_idx} отсутствует.")
            break

        # Лог всех кадров
        processed_frame_indices.append(absolute_index)

        # Проверяем, нужно ли этот кадр обрабатывать
        # if frame_idx % step == 0:
        if absolute_index % step == 0:
            frame_indices.append(absolute_index)
            # Рассчитываем абсолютное время кадра с учётом смещения сегмента
            timestamp_ms = int(frame_idx / round(fps, 0) * 1000) + segment_start
            # frame = cv2.resize(frame, (640, 360))

            logger.debug(f"Размер оригинального изображения {frame.shape=}")
            height_original, width_original = frame.shape[:2]
            # aspect_ratio_original = width_original / height_original

            # 1. Если высота кадра больше ширины
            if height_original > width_original:
                logger.debug("Анализируется вертикальное изображение")
                # Вычисляем ширину для соотношения 16:9
                new_width = int(height_original * 16 / 9)
                # Создаем черный холст с размерами (высота кадра, рассчитанная ширина)
                canvas = np.zeros((height_original, new_width, 3), dtype=np.uint8)
                # Вычисляем смещение для центровки кадра
                x_offset = (new_width - width_original) // 2
                # Накладываем кадр на черный холст
                canvas[:, x_offset : x_offset + width_original] = frame
                frame_extended = canvas
                logger.debug(
                    f"Размер изображения после "
                    f"стандартизации {frame_extended.shape=}"
                )
                frame_extended = cv2.resize(frame_extended, (360, 640))
                logger.debug(
                    f"Размер изображения после ресайза {frame_extended.shape=}"
                )
            else:
                frame_extended = frame

            frame_rgb = cv2.cvtColor(frame_extended, cv2.COLOR_BGR2RGB)

            processor.process_frame(frame_rgb, timestamp_ms)
            check_timestamps.append((frame_idx / round(fps, 0) * 1000) + segment_start)
            timestamps.append(timestamp_ms)
            processed_frames += 1

        frame_idx += 1

    # print(f"{segment_id=}, Все обработанные индексы кадров: {processed_frame_indices}")
    print(f"{segment_id=}, Обработано кадров: {len(frame_indices)}")
    print(
        f"{segment_id=}, Всего кадров: {total_frames}, Обработано кадров: {processed_frames}"
    )
    # print(f"{segment_id=}, Обработаны временные метки: {check_timestamps}")

    landmarks_data, world_landmarks_data, _ = processor.return_data()
    print(f"{segment_id=}, Форма landmarks_data: {landmarks_data.shape}")
    print(f"{segment_id=}, Форма world_landmarks_data: {world_landmarks_data.shape}")

    # Создаем временные файлы для хранения данных сегмента
    landmarks_file = os.path.join(output_dir, f"segment_{segment_id}_landmarks.npy")
    world_landmarks_file = os.path.join(
        output_dir, f"segment_{segment_id}_world_landmarks.npy"
    )
    timestamps_file = os.path.join(output_dir, f"segment_{segment_id}_timestamps.npy")

    # Сохраняем данные в файлы
    np.save(landmarks_file, landmarks_data)
    np.save(world_landmarks_file, world_landmarks_data)
    np.save(timestamps_file, np.array(timestamps, dtype=np.int64))
    print(f"{segment_id=}, Последняя обработанная метка времени: {timestamps[-1]}")

    # Для каждого сегмента сохраните первый и последний обработанный кадр
    print(
        f"Сегмент {segment_id}, Первый кадр: {processed_frame_indices[0]}, Последний кадр: {processed_frame_indices[-1]}"
    )

    indices_file = os.path.join(output_dir, f"segment_{segment_id}_indices.npy")
    np.save(indices_file, np.array(frame_indices, dtype=np.int64))

    cap.release()


def combine_segment_data(
    output_dir, combined_landmarks_file, combined_world_landmarks_file
):
    """
    Объединяет данные из всех сегментов в один файл, устраняя дублирующиеся кадры.

    :param output_dir: Директория с временными файлами данных.
    :param combined_landmarks_file: Путь к итоговому файлу для landmarks.
    :param combined_world_landmarks_file: Путь к итоговому файлу для world_landmarks.
    """

    def extract_segment_number(file_name):
        """
        Извлекает числовую часть из имени файла.
        Например: segment_10_landmarks.npy -> 10
        """
        return int(file_name.split("_")[1].split("_")[0])

    # Получаем файлы с данными landmarks и world_landmarks
    landmarks_files = sorted(
        [
            file_name
            for file_name in os.listdir(output_dir)
            if file_name.endswith("_landmarks.npy")
            and not file_name.endswith("_world_landmarks.npy")
        ],
        key=extract_segment_number,
    )

    world_landmarks_files = sorted(
        [
            file_name
            for file_name in os.listdir(output_dir)
            if file_name.endswith("_world_landmarks.npy")
        ],
        key=extract_segment_number,
    )

    timestamps_files = sorted(
        [
            file_name
            for file_name in os.listdir(output_dir)
            if file_name.endswith("_timestamps.npy")
        ],
        key=extract_segment_number,
    )

    indices_files = sorted(
        [
            file_name
            for file_name in os.listdir(output_dir)
            if file_name.endswith("_indices.npy")
        ],
        key=extract_segment_number,
    )

    print(f"Indices файлы для объединения: {indices_files}")
    print(f"Landmarks файлы для объединения: {landmarks_files}")
    print(f"World Landmarks файлы для объединения: {world_landmarks_files}")
    print(f"Timestamps файлы для объединения: {timestamps_files}")

    # Загружаем индексы
    all_frame_indices = []

    for indices_file in indices_files:
        indices = np.load(os.path.join(output_dir, indices_file))
        print(f"Файл {indices_file}: Индексов {len(indices)}")
        all_frame_indices.extend(indices)

    # Убираем дублирующиеся индексы
    unique_indices, unique_positions = np.unique(all_frame_indices, return_index=True)

    print(f"Всего уникальных индексов: {len(unique_indices)}")

    # Загружаем и объединяем данные
    all_landmarks_data = []
    all_world_landmarks_data = []
    all_timestamps = []

    # Проверка пересечений между метками времени
    for i in range(len(timestamps_files) - 1):
        current_timestamps = np.load(os.path.join(output_dir, timestamps_files[i]))
        next_timestamps = np.load(os.path.join(output_dir, timestamps_files[i + 1]))
        intersection = np.intersect1d(current_timestamps, next_timestamps)
        print(
            f"Пересечение меток времени между сегментами {i} и {i+1}: {len(intersection)}"
        )

    for landmarks_file, world_landmarks_file, timestamps_file in zip(
        landmarks_files, world_landmarks_files, timestamps_files
    ):
        landmarks = np.load(os.path.join(output_dir, landmarks_file))
        world_landmarks = np.load(os.path.join(output_dir, world_landmarks_file))
        timestamps = np.load(os.path.join(output_dir, timestamps_file))

        print(
            f"Файл {landmarks_file}: Кадров {len(landmarks)}, "
            f"Файл {world_landmarks_file}: Кадров {len(world_landmarks)}, "
            f"Метки времени: {len(timestamps)}"
        )

        all_landmarks_data.append(landmarks)
        all_world_landmarks_data.append(world_landmarks)
        all_timestamps.extend(timestamps)

    print(
        f"Всего уникальных меток времени после объединения: {len(np.unique(all_timestamps))}"
    )

    # Преобразуем в массивы
    all_landmarks_data = np.concatenate(all_landmarks_data, axis=0)
    all_world_landmarks_data = np.concatenate(all_world_landmarks_data, axis=0)
    all_timestamps = np.array(all_timestamps)

    # # Убираем дубли на основе меток времени
    # _, unique_indices = np.unique(all_timestamps, return_index=True)
    # all_landmarks_data = all_landmarks_data[unique_indices]
    # all_world_landmarks_data = all_world_landmarks_data[unique_indices]

    # # Сохраняем только уникальные кадры
    # all_landmarks_data = all_landmarks_data[unique_positions]
    # all_world_landmarks_data = all_world_landmarks_data[unique_positions]

    # Сохраняем объединённые данные
    np.save(combined_landmarks_file, all_landmarks_data)
    np.save(combined_world_landmarks_file, all_world_landmarks_data)

    # Удаляем временные файлы
    for file in landmarks_files + world_landmarks_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):  # Проверяем существование перед удалением
            os.remove(file_path)
            print(f"Удален временный файл: {file_path}")
        else:
            print(f"Файл уже был удалён: {file_path}")

    print(
        f"Все данные объединены в {combined_landmarks_file} и {combined_world_landmarks_file}"
    )


def clean_up_temp_files(segment_paths):
    """
    Удаление временных файлов.

    :param segment_paths: Список путей к временным файлам.
    """
    for path in segment_paths:
        try:
            os.remove(path)
        except OSError as e:
            print(f"Ошибка при удалении {path}: {e}")


def process_landmarks(video_file, video_hash):
    output_dir = f"app/resources/tmp/{video_hash}"
    os.makedirs(output_dir, exist_ok=True)
    combined_landmarks_file = f"{output_dir}/final_landmarks.npy"
    combined_world_landmarks_file = f"{output_dir}/final_world_landmarks.npy"

    segments_num = multiprocessing.cpu_count()

    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # step = 3
    step = int(fps / 8.33)

    clear_output_dir(output_dir)

    # Разделение видео
    start_split = time.time()
    segment_paths = split_video(video_file, segments_num, output_dir)
    logger.debug(f"Разделение видео завершено за {time.time() - start_split:.2f} сек.")

    # Рассчитываем смещения времени для каждого сегмента
    result = subprocess.run(
        [
            "ffprobe",
            "-i",
            video_file,
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=p=0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = float(result.stdout.strip())
    segment_duration = duration / segments_num
    segment_starts = [int(i * segment_duration * 1000) for i in range(segments_num)]

    # Обработка видео
    start_processing = time.time()
    with multiprocessing.Pool(processes=segments_num) as pool:
        pool.starmap(
            process_video_segment,
            [
                (path, step, start, output_dir)
                for path, start in zip(segment_paths, segment_starts)
            ],
        )
    print(
        f"Обработка всех сегментов завершена за {time.time() - start_processing:.2f} сек."
    )

    # Объединение данных
    combine_segment_data(
        output_dir, combined_landmarks_file, combined_world_landmarks_file
    )

    # Удаление временных файлов
    clean_up_temp_files(segment_paths)

    # Загрузка данных
    landmarks_data = np.load(combined_landmarks_file)
    world_landmarks_data = np.load(combined_world_landmarks_file)

    return landmarks_data, world_landmarks_data, None, step
