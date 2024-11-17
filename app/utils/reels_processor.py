import os
import shutil
import subprocess
import tempfile
import time
from collections import deque

import cv2  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
from utils.logger import setup_logger

# Инициализируем логгер
logger = setup_logger("main", "app/resources/logs/main.log")


class ReelsProcessor:
    def __init__(self, input_video, step=1):
        """
        Инициализация процессора видео.

        Параметры:
        - input_video: Путь к исходному видео.
        - video_fps: Частота кадров видео (по умолчанию 25).
        """
        self.input_video = input_video
        self.video_fps = self._get_video_fps()
        self.temp_files = []
        self.temp_dir = tempfile.mkdtemp()  # Временная директория для кадров
        self.step = step
        self.skeleton_connections = [
            (0, 2),
            (0, 5),
            (2, 7),
            (5, 8),
            (5, 4),
            (5, 6),
            (2, 1),
            (2, 3),
            (10, 9),
            (11, 12),
            (12, 14),
            (14, 16),
            (16, 22),
            (16, 20),
            (20, 18),
            (18, 16),
            (11, 13),
            (13, 15),
            (15, 21),
            (15, 19),
            (19, 17),
            (17, 15),
            (12, 24),
            (11, 23),
            (23, 24),
            (24, 26),
            (26, 28),
            (28, 30),
            (30, 32),
            (32, 28),
            (23, 25),
            (25, 27),
            (27, 29),
            (29, 31),
            (31, 27),
        ]

    def _get_video_fps(self):
        """
        Получает частоту кадров видео.
        """
        cap = cv2.VideoCapture(self.input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0:
            raise ValueError("Не удалось определить FPS видео")
        print(f"Частота кадров (FPS) видео: {fps}")
        return fps

    def interpolate_landmarks(self, landmarks_tensor):
        """
        Интерполирует нулевые значения в landmarks_tensor, усредняя по соседним значениям.
        """
        num_frames, num_points, num_coords = landmarks_tensor.shape

        for frame_idx in tqdm(range(num_frames)):
            for point_idx in range(num_points):
                # Проверяем, если все три координаты точки равны 0
                if np.array_equal(landmarks_tensor[frame_idx, point_idx], [0, 0, 0]):
                    prev_frame_idx = frame_idx - 1
                    next_frame_idx = frame_idx + 1

                    # Найти ближайшие предыдущий и следующий кадры с ненулевыми координатами для этой точки
                    while prev_frame_idx >= 0 and np.array_equal(
                        landmarks_tensor[prev_frame_idx, point_idx], [0, 0, 0]
                    ):
                        prev_frame_idx -= 1
                    while next_frame_idx < num_frames and np.array_equal(
                        landmarks_tensor[next_frame_idx, point_idx], [0, 0, 0]
                    ):
                        next_frame_idx += 1

                    # Если найдены валидные предыдущий и следующий кадры, усредняем
                    if prev_frame_idx >= 0 and next_frame_idx < num_frames:
                        landmarks_tensor[frame_idx, point_idx] = (
                            landmarks_tensor[prev_frame_idx, point_idx]
                            + landmarks_tensor[next_frame_idx, point_idx]
                        ) / 2
                    elif (
                        prev_frame_idx >= 0
                    ):  # Если есть только предыдущий, используем его координаты
                        landmarks_tensor[frame_idx, point_idx] = landmarks_tensor[
                            prev_frame_idx, point_idx
                        ]
                    elif (
                        next_frame_idx < num_frames
                    ):  # Если есть только следующий, используем его координаты
                        landmarks_tensor[frame_idx, point_idx] = landmarks_tensor[
                            next_frame_idx, point_idx
                        ]

        return landmarks_tensor

    def draw_trajectory(
        self,
        frame,
        center_points,
        point_color=(0, 0, 255),
        line_color=(0, 255, 0),
        interpolation_factor=1,
    ):
        """
        Рисует центральную точку и сглаженную траекторию на кадре.

        Параметры:
        - frame: текущий кадр видео.
        - center_points: список координат центральной точки на каждом кадре.
        - point_color: цвет центральной точки в формате BGR (по умолчанию красный).
        - line_color: цвет линии траектории в формате BGR (по умолчанию зеленый).
        - interpolation_factor: фактор интерполяции для создания дополнительных точек.

        Возвращает:
        - frame: кадр с нарисованной траекторией.
        """
        if len(center_points) > 1:
            # Разделение координат x и y для интерполяции
            x_points = [p[0] for p in center_points]
            y_points = [p[1] for p in center_points]

            # Создаем параметрическое представление данных для интерполяции
            t = np.arange(len(center_points))
            t_interpolated = np.linspace(
                0, len(center_points) - 1, len(center_points) * interpolation_factor
            )

            # Интерполяция координат x и y
            x_smooth = np.interp(t_interpolated, t, x_points)
            y_smooth = np.interp(t_interpolated, t, y_points)

            # Формируем сглаженные точки в формате, подходящем для cv2.polylines
            smooth_points = np.array(
                [[[int(x), int(y)] for x, y in zip(x_smooth, y_smooth)]], dtype=np.int32
            )

            # Рисуем сглаженную траекторию
            cv2.polylines(
                frame, smooth_points, isClosed=False, color=line_color, thickness=2
            )

        # Рисуем центральные точки для наглядности
        for point in center_points:
            cv2.circle(frame, point, 5, point_color, -1)  # Рисуем центральную точку

        return frame

    def draw_skeleton(
        self,
        frame,
        joints,
        point_color=(0, 255, 0),
        line_color=(255, 0, 0),
    ):
        """
        Рисует скелет, соединяя точки суставов в кадре.

        Параметры:
        - frame: текущий кадр видео.
        - joints: тензор координат суставов для одного кадра, размер (33, 3), где joints[i] = (x, y, z) для сустава i.
        - connections: список кортежей, указывающих, какие суставы нужно соединить.
        - point_color: цвет точек суставов в формате BGR.
        - line_color: цвет линий, соединяющих суставы, в формате BGR.

        Возвращает:
        - frame: кадр с нарисованным скелетом.
        """
        height, width = frame.shape[:2]  # Получаем размеры кадра

        # Преобразуем тензор в формат NumPy, оставляем только координаты x и y и масштабируем их
        joints_np = joints[:, :2].astype(int)

        # Отрисовка соединений между суставами
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx < len(joints_np) and end_idx < len(joints_np):
                start_point = tuple(joints_np[start_idx])
                end_point = tuple(joints_np[end_idx])
                # Рисуем линию между суставами
                cv2.line(frame, start_point, end_point, line_color, 2)

        # Отрисовка точек суставов
        for joint in joints_np:
            cv2.circle(frame, tuple(joint), 5, point_color, -1)

        return frame

    def fit_to_aspect_ratio_9_16(self, frame, crop_frame):
        """
        Вписывает обрезанное видео в фон 9:16, заполняя сверху и снизу размытым видео.

        Parameters:
        - frame: исходный кадр видео.
        - crop_frame: обрезанный кадр с паддингом.

        Returns:
        - output_frame: кадр 9:16 с размытым фоном.
        """
        # Размеры целевого фона (9:16)
        target_width = crop_frame.shape[1]
        target_height = int(target_width * (16 / 9))

        # Изменяем размер фона
        blurred_background = cv2.resize(frame, (target_width, target_height))
        blurred_background = cv2.GaussianBlur(blurred_background, (51, 51), 0)

        # Позиционируем обрезанный кадр по центру на фоне 9:16
        y_offset = (target_height - crop_frame.shape[0]) // 2
        output_frame = blurred_background.copy()
        output_frame[y_offset : y_offset + crop_frame.shape[0], :] = crop_frame

        # Проверяем размеры на кратность 2 и корректируем, если необходимо
        height, width = output_frame.shape[:2]
        if height % 2 != 0 or width % 2 != 0:
            new_height = height if height % 2 == 0 else height + 1
            new_width = width if width % 2 == 0 else width + 1
            output_frame = cv2.resize(output_frame, (new_width, new_height))

        return output_frame

    def process_jumps(
        self,
        jump_frames,
        landmarks_tensor,
        smooth_window=3,
        padding=0,
        draw_mode="Trajectory",
        progress=None,
    ):
        print("{Enter to ReelsProcessor}")
        # Применяем интерполяцию для замены нулевых значений в landmarks_tensor
        landmarks_tensor = self.interpolate_landmarks(landmarks_tensor)

        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_check = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{frames_check=}")

        # Устанавливаем размеры для кропа с соотношением 9:16 и делаем их четными
        crop_width = min(width, int(height * (9 / 16))) + 2 * padding
        crop_height = min(height, int(width * (16 / 9))) + 2 * padding

        # Округляем до ближайших четных чисел, чтобы соответствовать требованиям кодека
        crop_width = crop_width if crop_width % 2 == 0 else crop_width - 1
        crop_height = crop_height if crop_height % 2 == 0 else crop_height - 1

        center_points = []
        hand_points = []

        # Дек для хранения последних координат центральной точки для сглаживания
        recent_centers = deque(maxlen=smooth_window)

        original_bitrate = self.get_video_bitrate(self.input_video)
        processed_clips = []  # для хранения путей к обработанным клипам

        for idx, (start_frame, end_frame) in enumerate(jump_frames):
            logger.info(f"Отрисовка прыжка {idx+1}.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            # print(f"{landmarks_tensor.shape=}")
            frames_dir = os.path.join(self.temp_dir, f"jump_clip_{idx}")
            os.makedirs(frames_dir, exist_ok=True)

            # Сбрасываем счётчик для новой папки
            frame_count = 0

            print(f"Сохраняем кадры в {frames_dir}")
            for frame_idx in tqdm(
                range(start_frame, end_frame + 1),
                desc="Отрисовка видео",
            ):
                # print(f"{frame_idx=}")
                ret, frame = cap.read()
                if not ret:
                    break

                # Определяем, использовать ли реальные координаты из landmarks_tensor или усредненные
                if (frame_idx - start_frame) % self.step == 0:
                    # print(f"STEP FRAME - {frame_idx=}")
                    # print(f"LM IDX - {frame_idx // self.step=}")
                    # Индекс landmarks_tensor для текущего кадра
                    # Получаем исходные координаты центральной точки из landmarks_tensor
                    original_center_x = int(
                        (
                            landmarks_tensor[frame_idx // self.step, 23, 0]
                            + landmarks_tensor[frame_idx // self.step, 24, 0]
                        )
                        / 2
                        * width
                    )
                    original_center_y = int(
                        (
                            landmarks_tensor[frame_idx // self.step, 23, 1]
                            + landmarks_tensor[frame_idx // self.step, 24, 1]
                        )
                        / 2
                        * height
                    )
                    center_points.append((original_center_x, original_center_y))

                    # Добавляем координаты центральной точки в очередь для сглаживания
                    recent_centers.append((original_center_x, original_center_y))

                    # Вычисляем усредненные координаты центра для сглаживания
                    avg_center_x = int(np.mean([pt[0] for pt in recent_centers]))
                    avg_center_y = int(np.mean([pt[1] for pt in recent_centers]))

                    # Рассчитываем координаты для обрезки вокруг усредненного центра
                    x1 = max(0, min(avg_center_x - crop_width // 2, width - crop_width))
                    y1 = max(
                        0,
                        min(avg_center_y - crop_height // 2, height - crop_height),
                    )
                    x2 = x1 + crop_width
                    y2 = y1 + crop_height

                    # Выполняем кроп кадра
                    cropped_frame = frame[y1:y2, x1:x2]

                    # Обновляем список центральных точек для кропнутого кадра
                    cropped_center_points = [(x - x1, y - y1) for x, y in center_points]

                    # Получаем исходные координаты руки и корректируем их относительно кропа
                    original_hand_x = int(
                        landmarks_tensor[frame_idx // self.step, 0, 0] * width
                    )
                    original_hand_y = int(
                        landmarks_tensor[frame_idx // self.step, 0, 1] * height
                    )
                    hand_points.append((original_hand_x, original_hand_y))

                    # Обновляем список точек руки для кропнутого кадра
                    cropped_hand_points = [(x - x1, y - y1) for x, y in hand_points]

                    if draw_mode == "Trajectory":
                        # Рисуем центральную точку и траекторию на кропнутом кадре
                        cropped_frame = self.draw_trajectory(
                            cropped_frame,
                            cropped_center_points,
                            point_color=(102, 153, 0),
                            line_color=(102, 153, 0),
                        )

                        # Рисуем траекторию руки на кропнутом кадре
                        cropped_frame = self.draw_trajectory(
                            cropped_frame,
                            cropped_hand_points,
                            point_color=(0, 102, 153),
                            line_color=(0, 102, 153),
                        )
                    elif draw_mode == "Skeleton":
                        # Пересчитываем координаты суставов для кропнутого кадра
                        joints = landmarks_tensor[
                            frame_idx // self.step, :, :2
                        ] * np.array([width, height])
                        cropped_joints = np.array(
                            [(int(x - x1), int(y - y1)) for x, y in joints]
                        )
                        cropped_frame = self.draw_skeleton(
                            cropped_frame, cropped_joints
                        )

                    # Вписываем обрезанный кадр в размытую версию
                    if padding != 0:
                        output_frame = self.fit_to_aspect_ratio_9_16(
                            frame, cropped_frame
                        )
                    else:
                        output_frame = cropped_frame

                # Если кадр не кратен step, используем усредненные координаты из recent_centers
                else:
                    if recent_centers:
                        avg_center_x = int(np.mean([pt[0] for pt in recent_centers]))
                        avg_center_y = int(np.mean([pt[1] for pt in recent_centers]))
                    else:
                        avg_center_x, avg_center_y = (
                            width // 2,
                            height // 2,
                        )  # центральная точка по умолчанию

                    # Рассчитываем координаты для обрезки вокруг усредненного центра
                    x1 = max(0, min(avg_center_x - crop_width // 2, width - crop_width))
                    y1 = max(
                        0, min(avg_center_y - crop_height // 2, height - crop_height)
                    )
                    x2 = x1 + crop_width
                    y2 = y1 + crop_height

                    # Выполняем кроп кадра
                    cropped_frame = frame[y1:y2, x1:x2]

                    # Вписываем обрезанный кадр в размытую версию
                    if padding != 0:
                        output_frame = self.fit_to_aspect_ratio_9_16(
                            frame, cropped_frame
                        )
                    else:
                        output_frame = cropped_frame

                # Сохраняем кропнутый кадр в виде изображения
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.png")
                # cv2.imwrite(frame_path, output_frame)
                if not cv2.imwrite(frame_path, output_frame):
                    print(f"Не удалось сохранить кадр {frame_count} в {frame_path}")
                frame_count += 1

            time.sleep(1)
            logger.info(f"Создание видео из кадров прыжка {idx+1}.")
            # Создание видео из кропнутых кадров с помощью ffmpeg
            clip_path = os.path.join(self.temp_dir, f"jump_clip_{idx}.mp4")
            # Путь к файлу для записи логов ffmpeg
            ffmpeg_log_path = "ffmpeg_concat.log"
            # Открываем файл для записи логов
            with open(ffmpeg_log_path, "a") as log_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-framerate",
                        str(self.video_fps),
                        "-i",
                        os.path.join(frames_dir, "frame_%05d.png"),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-b:v",
                        f"{original_bitrate}",  # Применяем оригинальный битрейт
                        "-profile:v",
                        "high",  # Используем профиль высокого качества
                        "-crf",
                        "18",  # Улучшаем качество путем настройки компрессии
                        clip_path,
                    ],
                    stdout=log_file,
                    stderr=log_file,
                )

            processed_clips.append(clip_path)

        cap.release()

        # Объединяем все фрагменты в итоговое видео
        final_output_path = os.path.join("processed_video_with_fades.mp4")
        self.concat_clips(processed_clips, final_output_path)

        # Очистка временных файлов
        self.cleanup_temp_files()

        return final_output_path

    # Дополнительная функция для получения длительности видео
    def get_video_duration(self, video_path):
        """Возвращает длительность видео в секундах."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())

    def get_video_bitrate(self, video_path):
        """Возвращает битрейт видео в формате kbps."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())

    def concat_clips(self, clips, output_path):
        # Путь к файлу для записи логов ffmpeg
        ffmpeg_log_path = "ffmpeg_concat.log"

        # Открываем файл для записи логов
        with open(ffmpeg_log_path, "a") as log_file:
            # Временные параметры для эффектов
            intermediate_output = "intermediate_temp.mp4"
            fade_duration = 1  # Длительность переходов
            original_bitrate = self.get_video_bitrate(self.input_video)

            logger.info("Создание вступительного перехода видео")
            # Применяем fade-in к первому клипу
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    clips[0],
                    "-vf",
                    f"fade=t=in:st=0:d={fade_duration}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    f"{original_bitrate}",  # Применяем оригинальный битрейт
                    "-profile:v",
                    "high",
                    "-crf",
                    "18",
                    intermediate_output,
                ],
                stdout=log_file,
                stderr=log_file,
            )

            # Начальный offset для первого перехода
            current_offset = (
                self.get_video_duration(intermediate_output) - fade_duration
            )

            logger.info("Создание переходов между прыжками")
            # Проходим по остальным клипам, объединяя их с эффектом xfade и корректируя offset
            for i in range(1, len(clips)):
                next_clip = clips[i]
                xfade_output = f"temp_xfade_{i}.mp4"

                logger.info(f"... прыжок {i}")
                # Склеиваем текущий промежуточный файл с очередным клипом
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        intermediate_output,
                        "-i",
                        next_clip,
                        "-filter_complex",
                        f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={current_offset}[v]",
                        "-map",
                        "[v]",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-b:v",
                        f"{original_bitrate}",  # Применяем оригинальный битрейт
                        "-profile:v",
                        "high",
                        "-crf",
                        "18",
                        xfade_output,
                    ],
                    stdout=log_file,
                    stderr=log_file,
                )

                # Обновляем промежуточный файл
                intermediate_output = xfade_output

                # Обновляем offset для следующего перехода
                current_offset += self.get_video_duration(next_clip) - fade_duration

            logger.info("Создание заключительного перехода видео")
            # Добавляем fade-out к последнему объединенному клипу
            final_duration = self.get_video_duration(intermediate_output)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    intermediate_output,
                    "-vf",
                    f"fade=t=out:st={final_duration - fade_duration}:d={fade_duration}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    f"{original_bitrate}",  # Применяем оригинальный битрейт
                    "-profile:v",
                    "high",
                    "-crf",
                    "18",
                    output_path,
                ],
                stdout=log_file,
                stderr=log_file,
            )

        logger.info("Готово")
        # Удаляем временные файлы
        for i in range(1, len(clips)):
            os.remove(f"temp_xfade_{i}.mp4")
        if os.path.exists(intermediate_output):
            os.remove(intermediate_output)

    def cleanup_temp_files(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_files.clear()
