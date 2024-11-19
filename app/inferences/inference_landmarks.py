import gc

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class LandmarksProcessor:
    def __init__(
        self,
        model_path,
        key,
        new_width=640,
        new_height=360,
        display=False,
        calculate_masks=False,
        do_resize=False,
    ):
        self.model_path = model_path
        self.new_width = new_width
        self.new_height = new_height
        self.width = None
        self.height = None
        self.display = display
        self.calculate_masks = calculate_masks
        self.do_resize = do_resize
        # Буфер для сохранения данных
        self.buffer = []
        # Инициализация детектора
        self._initialize_detector()
        self.key = key

    def _initialize_detector(self):
        """Инициализация или повторная инициализация детектора."""
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=self.calculate_masks,  # Включаем или отключаем расчет масок сегментации
        )
        self.detector = PoseLandmarker.create_from_options(self.options)

    def process_frame(self, frame, timestamp_ms):
        # Обработка кадра (при необходимости изменяем его размер)
        # Здесь также уйду от обработки OpenCV - занимает немало времени
        if self.do_resize:
            frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
        else:
            frame_resized = frame

        self.width = frame_resized.shape[0]
        self.height = frame_resized.shape[1]
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Обработка кадра и получение результата синхронно
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        self.result_callback(detection_result, timestamp_ms)

    def result_callback(self, detection_result, timestamp_ms):
        """Обработка результатов детекции для каждого кадра."""
        pose_landmarks_list = detection_result.pose_landmarks
        pose_world_landmarks_list = detection_result.pose_world_landmarks
        segmentation_masks = (
            detection_result.segmentation_masks if self.calculate_masks else None
        )

        if pose_landmarks_list and pose_world_landmarks_list:
            for pose_landmarks, pose_world_landmarks in zip(
                pose_landmarks_list, pose_world_landmarks_list
            ):
                landmarks_array = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in pose_landmarks
                    ],
                    dtype=np.float16,
                )
                world_landmarks_array = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in pose_world_landmarks
                    ],
                    dtype=np.float16,
                )

                mask_array = (
                    segmentation_masks[0].numpy_view().astype(np.float16)
                    if self.calculate_masks
                    else None
                )

                # Сохраняем данные в буфер
                self.buffer.append(
                    (timestamp_ms, landmarks_array, world_landmarks_array, mask_array)
                )
        else:
            landmarks_array = np.zeros((33, 3), dtype=np.float16)
            world_landmarks_array = np.zeros((33, 3), dtype=np.float16)
            mask_array = (
                np.zeros((self.width, self.height), dtype=np.float16)
                if self.calculate_masks
                else None
            )

            # Сохраняем пустые данные в буфер
            self.buffer.append(
                (timestamp_ms, landmarks_array, world_landmarks_array, mask_array)
            )

    def return_data(self):
        """Возвращает собранные данные из буфера."""
        if self.buffer:
            landmarks_data = np.array([item[1] for item in self.buffer])
            world_landmarks_data = np.array([item[2] for item in self.buffer])
            masks_data = (
                np.array([item[3] for item in self.buffer if item[3] is not None])
                if self.calculate_masks
                else None
            )
            self.buffer.clear()

            return (landmarks_data, world_landmarks_data, masks_data)

    def process_video(self, video_path, calculate_type):
        """
        Здесь планируется уйти от OpenCV и не читать все кадры подряд.
        Планирую работать напрямую с ffmpeg
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть видео.")
            return None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if calculate_type == "pre":
            step = int(fps / 8.33)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{total_frames=}")
        total_processed_frames = total_frames // step
        batch_size = 50
        frames_batch = []
        check_ids = []

        # Добавляем прогресс-бар с общим количеством кадров
        with tqdm(
            total=total_processed_frames,
            desc="Анализ видео",
        ) as pbar:
            frame_idx = 0
            processed_frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Проверяем, нужно ли этот кадр обрабатывать
                if frame_idx % step == 0:
                    frames_batch.append(frame)
                    processed_frame_count += 1
                    check_ids.append(frame_idx)

                if len(frames_batch) == batch_size:
                    for i, frame in enumerate(frames_batch):
                        timestamp_ms = int(
                            ((frame_idx - len(frames_batch) + i + 1) / fps) * 1000
                        )
                        self.process_frame(frame, timestamp_ms)
                        pbar.update(1)
                    frames_batch.clear()
                    check_ids = []
                frame_idx += 1

            # Обрабатываем оставшиеся кадры в конце
            if frames_batch:
                for frame in frames_batch:
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    self.process_frame(frame, timestamp_ms)
                    check_ids.append(frame_idx)
                    frame_idx += 1
                    pbar.update(1)  # Обновляем прогресс-бар

        cap.release()
        landmarks_data, world_landmarks_data, masks_data = self.return_data()
        self.cleanup()

        return landmarks_data, world_landmarks_data, masks_data

    def cleanup(self):
        """Метод для очистки и освобождения ресурсов."""
        self.buffer.clear()  # Очищаем буфер данных
        del self.detector  # Удаляем детектор
        gc.collect()  # Принудительный запуск сборщика мусора для освобождения памяти
