import os
import pickle
import shutil
import subprocess
import tempfile
import time

# from collections import deque
import cv2  # type: ignore
import numpy as np  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from tqdm import tqdm  # type: ignore
from utils.logger import setup_logger

# Инициализируем логгер
logger = setup_logger("main", "app/resources/logs/main.log")

CLIP_DIR = "app/resources/outputs"
os.makedirs(CLIP_DIR, exist_ok=True)
CROPS_DIR = "app/resources/crops_cache"
os.makedirs(CROPS_DIR, exist_ok=True)


class SmoothWindowTracker:
    """
    Класс для плавного отслеживания положения объекта с фильтрацией.
    """

    def __init__(self, initial_x, alpha=0.3, threshold=10):
        """
        Инициализация трекера.

        :param initial_x: Начальная позиция
        :param alpha: Коэффициент сглаживания (по умолчанию 0.3)
        :param threshold: Пороговое значение изменения позиции (по умолчанию 10)
        """
        self.current_x = initial_x
        self.alpha = alpha
        self.threshold = threshold

    def __call__(self, detected_x):
        """
        Вызывается для обновления текущей позиции с фильтрацией.

        :param detected_x: Новая обнаруженная позиция
        :return: Отфильтрованная позиция
        """
        if abs(detected_x - self.current_x) > self.threshold:
            self.current_x = self.alpha * detected_x + (1 - self.alpha) * self.current_x
        return self.current_x


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

    def interpolate_zero_landmarks(self, landmarks):
        """
        Интерполирует нулевые значения landmarks с использованием линейной интерполяции.

        :param landmarks: Массив формы (num_frames, num_points, coord_dims), где
                        num_frames — количество кадров,
                        num_points — количество точек,
                        coord_dims — размерность координат (например, 2D или 3D).
        :type landmarks: numpy.ndarray
        :return: Массив той же формы, с интерполированными координатами.
        :rtype: numpy.ndarray
        """
        interpolated_landmarks = landmarks.copy()
        num_frames, num_points, coord_dims = landmarks.shape

        for point_idx in range(num_points):
            for dim in range(coord_dims):
                # Извлечение значений для данной точки и размерности
                values = landmarks[:, point_idx, dim]
                frames = np.arange(num_frames)

                # Проверяем, какие кадры имеют ненулевые значения
                non_zero_mask = values != 0
                non_zero_frames = frames[non_zero_mask]
                non_zero_values = values[non_zero_mask]

                if len(non_zero_frames) < 2:
                    # Если недостаточно точек для интерполяции, пропускаем
                    continue

                # Линейная интерполяция для заполнения нулей
                interpolator = interp1d(
                    non_zero_frames,
                    non_zero_values,
                    kind="linear",
                    fill_value="extrapolate",
                )
                interpolated_values = interpolator(frames)

                # Заполняем нули интерполированными значениями
                interpolated_landmarks[:, point_idx, dim] = interpolated_values

        return interpolated_landmarks

    def interpolate_step_landmarks(self, landmarks, step=3):
        """
        Интерполирует координаты landmarks между кадрами.

        :param landmarks: Входной массив с формой (N, 33, 3), где N - количество кадров.
        :param step: Шаг между кадрами, для которых известны координаты.
        :return: Расширенный массив с интерполированными значениями.
        """
        num_frames = (landmarks.shape[0] - 1) * step + 1
        num_landmarks, num_coords = landmarks.shape[1], landmarks.shape[2]

        # Индексы известных кадров
        known_frames = np.arange(0, num_frames, step)
        # Индексы всех кадров
        all_frames = np.arange(0, num_frames)

        # Результирующий массив
        interpolated = np.zeros((num_frames, num_landmarks, num_coords))

        for landmark_idx in range(num_landmarks):
            for coord_idx in range(num_coords):
                # Линейная интерполяция
                interpolated[:, landmark_idx, coord_idx] = np.interp(
                    all_frames, known_frames, landmarks[:, landmark_idx, coord_idx]
                )

        return interpolated

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

    def get_figure_bbox(self, landmarks, frames, padding=0.05):
        """
        Рассчитывает рамку для фигуры в каждом кадре с отступом (padding).

        :param landmarks: Массив с координатами для каждого кадра
        :param frames: Список номеров кадров
        :param padding: Отступ для рамки (по умолчанию 0.05)
        :return: Массив рамок (bbox) для каждого кадра, размером (количество кадров, 2, 2)
                где bbox[i][0] содержит координаты (x_min, y_min) и bbox[i][1] содержит (x_max, y_max)
        """
        bbox = np.zeros((len(frames), 2, 2))  # Массив для хранения рамок

        for i, frame in enumerate(frames):
            x = landmarks[frame][:, 0]
            y = landmarks[frame][:, 1]

            # Расчет минимальных и максимальных значений с учетом отступа
            x_min, x_max = x.min() - padding, x.max() + padding
            y_min, y_max = y.min() - padding, y.max() + padding

            # Заполнение bbox для текущего кадра
            bbox[i][0] = [x_min, y_min]
            bbox[i][1] = [x_max, y_max]

        return bbox

    def get_max_width(self, bboxes):
        """
        Находит максимальную ширину среди всех рамок (bbox).

        :param bboxes: Массив рамок, где каждая рамка имеет формат [[x_min, y_min], [x_max, y_max]]
        :return: Максимальная ширина среди всех рамок
        """
        max_width = 0
        for bbox in bboxes:
            x_min, _ = bbox[0]
            x_max, _ = bbox[1]
            width = x_max - x_min
            if width > max_width:
                max_width = width
        return max_width

    def process_jumps(
        self,
        jump_frames,
        landmarks_tensor,
        smooth_window=3,
        padding=0,
        draw_mode="Trajectory",
        progress=None,
        video_hash=None,
    ):
        # Применяем интерполяцию для замены нулевых значений в landmarks_tensor
        # и формирования данных для всего видео с шагом 1
        landmarks = self.interpolate_zero_landmarks(landmarks_tensor)
        landmarks = self.interpolate_step_landmarks(landmarks, step=self.step)

        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        resolution_original_video = (width, height)  # (w, h)
        padding = 0.05

        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {self.input_video}")

        frames_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(frames_dir, exist_ok=True)
        clip_dir = os.path.join(CLIP_DIR, video_hash)
        os.makedirs(clip_dir, exist_ok=True)
        crops_file = os.path.join(CROPS_DIR, f"{video_hash}.pkl")

        full_frames = []
        for start_frame, end_frame in jump_frames:
            full_frames += list(range(start_frame, end_frame))

        # Рассчитываем рамки фигур
        bboxes = self.get_figure_bbox(landmarks, full_frames, padding=padding)
        # Фиксируем вертикальную позицию rect_reel
        max_width = self.get_max_width(bboxes)
        max_height = max_width * (16 / 9) * (16 / 9)
        fixed_y_reel = -0.5 - max_height / 2

        max_width = int(max_width * resolution_original_video[0])
        if max_width % 2 != 0:
            max_width += 1

        max_height = int(max_height * resolution_original_video[1])
        if max_height % 2 != 0:
            max_height += 1

        fixed_y_reel *= resolution_original_video[1]

        logger.debug(f"{max_width=}")
        logger.debug(f"{max_height=}")
        logger.debug(f"{fixed_y_reel=}")

        fade_points = []
        frame_count = 0
        for idx, (start_frame, end_frame) in enumerate(jump_frames):
            logger.debug(f"--- Прыжок {idx=} ---")
            frames = list(range(start_frame, end_frame))
            logger.debug(f"{start_frame=}, {end_frame=}")

            # Если end_frame не задан, установить его на последний кадр видео
            if end_frame is None:
                end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Рассчитываем рамки фигур
            bboxes = self.get_figure_bbox(landmarks, frames, padding=padding)

            bboxes[:, 0, 0] *= resolution_original_video[0]
            bboxes[:, 1, 0] *= resolution_original_video[0]
            bboxes[:, 0, 1] *= resolution_original_video[1]
            bboxes[:, 1, 1] *= resolution_original_video[1]

            x_min, _ = bboxes[0][0]
            x_max, _ = bboxes[0][1]
            initial_x = (x_max + x_min) / 2 - max_width / 2
            swt_filter = SmoothWindowTracker(
                initial_x=initial_x,
                alpha=0.25,
                threshold=0.025 * resolution_original_video[0],
            )
            x_filter = swt_filter

            x_min, _ = bboxes[:, 0, 0], bboxes[:, 0, 1]
            x_max, _ = bboxes[:, 1, 0], bboxes[:, 1, 1]

            # Вычисляем `detected_x` для всех кадров
            detected_x = (x_max + x_min) / 2 - max_width / 2
            # Ограничение в пределах [0, 1 - max_width]
            detected_x = np.clip(
                detected_x, 0, resolution_original_video[0] - max_width
            )

            # Применяем фильтр, если он есть
            if x_filter:
                filtered_x = np.array([x_filter(dx) for dx in detected_x]).astype(int)
            else:
                filtered_x = detected_x.astype(int)

            crop_y_min = np.full_like(filtered_x, -fixed_y_reel)
            crop_y_max = np.full_like(filtered_x, -(fixed_y_reel + max_height))
            crop_y_min = np.clip(crop_y_min, 0, resolution_original_video[1])
            crop_y_max = np.clip(crop_y_max, 0, resolution_original_video[1])
            crop_y_min, crop_y_max = crop_y_max, crop_y_min

            crop_x_min = filtered_x
            crop_x_max = filtered_x + max_width
            crop_x_min = np.clip(crop_x_min, 0, resolution_original_video[0])
            crop_x_max = np.clip(crop_x_max, 0, resolution_original_video[0])

            crops = np.column_stack(
                [
                    crop_x_min,
                    crop_y_min,
                    np.full_like(crop_x_min, max_width),
                    np.full_like(crop_x_min, (crop_y_max - crop_y_min)),
                ]
            )

            with open(crops_file, "wb") as f:
                pickle.dump(crops, f)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_idx in tqdm(
                range(0, end_frame - start_frame),
                desc="Отрисовка видео",
            ):
                ret, frame = cap.read()
                if not ret:
                    break  # Достигнут конец видео

                # Выполняем кроп кадра
                x1 = crops[frame_idx][0]
                x2 = crops[frame_idx][0] + crops[frame_idx][2]
                y1 = crops[frame_idx][1]
                y2 = crops[frame_idx][1] + crops[frame_idx][3]
                cropped_frame = frame[y1:y2, x1:x2]

                # Сохраняем кропнутый кадр в виде изображения
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.png")
                if not cv2.imwrite(frame_path, cropped_frame):
                    print(f"Не удалось сохранить кадр {frame_idx} в {frame_path}")
                frame_count += 1

            logger.debug(f"{frame_count=}")
            fade_points.append(frame_count)

        logger.debug(f"{fade_points=}")
        del fade_points[-1]
        logger.debug(f"{fade_points=}")

        cap.release()

        # original_bitrate = self.get_video_bitrate(self.input_video)
        clip_path = os.path.join(clip_dir, "clip.mp4")
        # Путь к файлу для записи логов ffmpeg
        ffmpeg_log_path = f"{clip_dir}/ffmpeg.log"

        # Длительность эффекта fade-in и fade-out (в секундах)
        fade_duration = 1
        total_frames = len(
            [
                f
                for f in os.listdir(frames_dir)
                if f.startswith("frame_") and f.endswith(".png")
            ]
        )
        total_duration = total_frames / fps  # Общая длительность видео

        # Вычисляем временные метки переходов
        fade_points_sec = [point / fps for point in fade_points]

        filter_complex = f"[0:v]fps={fps},split={len(fade_points)+1}"

        # Формируем вступительную часть
        for i in range(len(fade_points) + 1):
            filter_complex += f"[v{i}]"

        filter_complex += ";"

        if len(fade_points) > 0:
            # Формируем части для `trim` и `fps`
            for i, (start, end) in enumerate(
                zip(
                    [0] + fade_points_sec,
                    fade_points_sec + [total_duration],
                )
            ):
                filter_complex += (
                    f"[v{i}]trim={start}:{end},setpts=PTS-STARTPTS[v{i}trim];"
                )
                filter_complex += f"[v{i}trim]fps={fps}[v{i}fixed];"

            # Формируем части для `xfade`
            for i in range(len(fade_points_sec)):
                offset = fade_points_sec[i] - (i + 1) * (fade_duration / 2)
                prev = f"vx{i}fade" if i > 0 else "v0fixed"
                filter_complex += f"[{prev}][v{i+1}fixed]xfade=transition=fade:duration={fade_duration}:offset={offset}[vx{i+1}fade];"

            filter_complex += f"[vx{i+1}fade]fade=in:0:{int(fade_duration * fps)},"
        else:
            filter_complex += f"[v0]fade=in:0:{int(fade_duration * fps)},"

        # Формируем части для `fade` и отступов
        filter_complex += (
            f"fade=out:{int((total_duration - (i+1)*(fade_duration / 2) - fade_duration) * fps)}:{int(fade_duration * fps)},"
            f"split[vmain][vblur];"
            f"[vblur]scale=iw:-1,boxblur=luma_radius=20:luma_power=1,"
            f"scale=iw:iw*16/9,setsar=1[vbg];"
            f"[vbg][vmain]overlay=0:(H-h)/2,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2[vout]"
        )

        # Сохраняем в файл
        filter_file = f"{clip_dir}/filter_complex.txt"
        with open(filter_file, "w") as f:
            f.write(filter_complex)

        time.sleep(1)
        # Запуск команды FFmpeg
        with open(ffmpeg_log_path, "a") as log_file:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    os.path.join(frames_dir, "frame_%05d.png"),
                    "-filter_complex",
                    # filter_file,
                    filter_complex,
                    "-map",
                    "[vout]",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    "4M",  # Примерно 4 Mbps
                    "-profile:v",
                    "high",
                    "-crf",
                    "18",
                    clip_path,
                ],
                stdout=log_file,
                stderr=log_file,
            )

        return clip_path

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
