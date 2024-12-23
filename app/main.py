import os
import queue
import re
import subprocess
import tempfile
import threading
import uuid

import cv2
import numpy as np
from inferences.inference_landmarks import LandmarksProcessor
from PIL import Image, ImageDraw, ImageFont
from utils import utils as u
from utils.logger import setup_logger
from utils.main_process import process_video

import gradio as gr

# Инициализируем логгер
log_file = "app/resources/logs/main.log"
logger = setup_logger("main", log_file)
hash_pattern = re.compile(r"/([a-f0-9]{64})/")

BUFFER_SIZE = 5  # Размер буфера (количество фрагментов)
CHUNK_DURATION = 5  # Длительность каждого фрагмента (в секундах)
SKELETON_CONNECTIONS = [
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


def enable_button(video_file):
    # Включаем кнопку только если видео загружено
    is_loaded = bool(video_file)
    if is_loaded:
        logger.debug("Видео загружено, разблокирована кнопка запуска")
        logger.debug(f"Загружено видео: {hash_pattern.search(video_file).group(1)}")
    else:
        logger.debug("Загруженное видео удалено, кнопка запуска заблокирована")
    return gr.update(interactive=is_loaded)


def draw_skeleton(
    frame,
    joints,
    skeleton_connections,
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
    joints_np = (joints[:, :2] * np.array([width, height])).astype(int)
    # print(f"Shape of joints: {joints.shape}")
    # print(f"Example joint: {joints[0, 0]}")
    # print(f"Shape of joints: {joints_np.shape}")
    # print(f"Example joint: {joints_np[0, 0]}")

    # Отрисовка соединений между суставами
    for start_idx, end_idx in skeleton_connections:
        if start_idx < len(joints_np) and end_idx < len(joints_np):
            start_point = tuple(joints_np[start_idx])
            end_point = tuple(joints_np[end_idx])
            # Рисуем линию между суставами
            cv2.line(frame, start_point, end_point, line_color, 2)

    # Отрисовка точек суставов
    for joint in joints_np:
        cv2.circle(frame, tuple(joint), 5, point_color, -1)

    return frame


def convert_to_h264_ts(input_file, video_hash):
    """
    Конвертирует видеофайл в .ts формат с кодеком H.264.
    Args:
        input_file (str): Путь к входному видеофайлу.
    Returns:
        str: Путь к выходному .ts файлу.
    """
    unique_id = uuid.uuid4().hex  # Уникальный идентификатор
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=f"app/resources/tmp/{video_hash}",
        suffix=f"_{unique_id}.ts",
    )
    output_file = temp_file.name
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-f",
        "mpegts",
        "-loglevel",
        "error",
        output_file,
    ]
    subprocess.run(command, check=True)
    return output_file


def darken_frame(frame, alpha=0.7):
    """
    Затемняет кадр.
    Args:
        frame: Исходный кадр.
        alpha: Коэффициент затемнения (0.0 - полностью черный, 1.0 - без изменений).
    Returns:
        Затемненный кадр.
    """
    return (frame * alpha).astype(np.uint8)


def add_text_with_pillow(
    frame,
    text="Идет анализ",
    font_path="arial.ttf",
    font_size=32,
    color=(255, 255, 255),
):
    """
    Добавляет текст на русском языке на изображение с использованием Pillow.
    Args:
        frame: Кадр в формате numpy (OpenCV).
        text: Текст для отображения.
        font_path: Путь к файлу шрифта (например, Arial).
        font_size: Размер шрифта.
        color: Цвет текста (RGB).
    Returns:
        frame: Кадр с наложенным текстом.
    """
    # Конвертируем OpenCV изображение в формат Pillow
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    # Загружаем шрифт
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise RuntimeError(f"Не удалось загрузить шрифт: {font_path}")

    # Определяем размеры текста
    width, height = image.size
    text_bbox = draw.textbbox(
        (0, 0), text, font=font
    )  # Возвращает координаты ограничивающего прямоугольника
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2

    # Наносим текст
    draw.text((text_x, text_y), text, font=font, fill=color)

    # Конвертируем обратно в OpenCV формат
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def add_loading_indicator(
    frame, frame_idx, radius=25, color=(0, 0, 255), speed=3, sector_size=30
):
    """
    Добавляет плавно изменяющийся анимированный индикатор загрузки (малый сектор).
    Args:
        frame: Исходный кадр.
        frame_idx: Индекс текущего кадра (для анимации).
        radius: Радиус круга.
        color: Цвет круга (BGR).
        speed: Скорость изменения угла (градусы на кадр).
        sector_size: Размер сектора в градусах.
    Returns:
        Кадр с наложенным индикатором.
    """
    height, width = frame.shape[:2]
    center = (width // 2, height // 2 + 150)  # Расположение индикатора чуть ниже текста

    # Плавно изменяющийся начальный угол
    start_angle = (frame_idx * speed) % 360

    # Конечный угол сектора
    end_angle = (start_angle + sector_size) % 360

    # Рисуем сектор
    if end_angle > start_angle:
        # Обычный случай: сектор не пересекает 0 градусов
        cv2.ellipse(
            frame, center, (radius, radius), 0, start_angle, end_angle, color, 5
        )
    else:
        # Случай, когда сектор пересекает 0 градусов
        cv2.ellipse(frame, center, (radius, radius), 0, start_angle, 360, color, 5)
        cv2.ellipse(frame, center, (radius, radius), 0, 0, end_angle, color, 5)

    return frame


def process_video_to_buffer(video_path, buffer_queue):
    """
    Обрабатывает видео и кладет готовые фрагменты в буфер.
    Args:
        video_path (str): Путь к исходному видеофайлу.
        buffer_queue (queue.Queue): Очередь для хранения готовых фрагментов.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        processor = LandmarksProcessor(
            "app/inferences/models/landmarkers/pose_landmarker_lite.task",
            "stream",
            do_resize=False,
        )

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        step = int((fps / 8.33))
        adjusted_fps = fps // step  # Скорректированная частота кадров
        video_hash = u.generate_video_hash(video_path)
        frame_idx = 0
        while cap.isOpened():
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                dir=f"app/resources/tmp/{video_hash}",
                suffix=".mp4",
            )
            temp_video_path = temp_file.name
            temp_file.close()

            # Создаем видеозапись для текущего фрагмента
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                temp_video_path, fourcc, adjusted_fps, (frame_width, frame_height)
            )

            for _ in range(int(fps * CHUNK_DURATION)):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    # if frame_idx % 1 == 0:
                    timestamp_ms = int(frame_idx / fps * 1000)
                    processor.process_frame(frame, timestamp_ms)
                    skeleton, _, _ = processor.return_data()
                    # logger.debug(f"{skeleton.shape=}")
                    # logger.debug(f"{skeleton=}")
                    # Затемняем кадр
                    frame = darken_frame(frame)

                    frame = add_text_with_pillow(
                        frame,
                        text="Идет анализ...",
                        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        font_size=32,
                        color=(255, 255, 255),
                    )

                    # Добавление индикатора загрузки
                    frame = add_loading_indicator(frame, frame_idx)

                    frame = draw_skeleton(
                        frame,
                        skeleton[0],
                        SKELETON_CONNECTIONS,
                        point_color=(0, 255, 0),
                        line_color=(255, 0, 0),
                    )
                    out.write(frame)
                frame_idx += 1

            out.release()

            if os.path.exists(temp_video_path):
                ts_file = convert_to_h264_ts(temp_video_path, video_hash)
                os.unlink(temp_video_path)  # Удаляем временный mp4 файл
                buffer_queue.put(ts_file)  # Кладем готовый файл в очередь

            if not ret:
                break

        cap.release()
        buffer_queue.put(None)  # Указатель на завершение обработки

    except Exception as e:
        logger.error(f"Ошибка во время обработки видео: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        buffer_queue.put(None)  # Завершающий маркер


def generate_stream_with_buffer(video_path):
    """
    Генерирует поток видео с использованием буфера.
    Args:
        video_path (str): Путь к исходному видеофайлу.
    Yields:
        str: Путь к обработанным видеофрагментам (.ts).
    """
    buffer_queue = queue.Queue(maxsize=BUFFER_SIZE)

    # Запускаем обработку видео в отдельном потоке
    threading.Thread(
        target=process_video_to_buffer, args=(video_path, buffer_queue), daemon=True
    ).start()

    while True:
        ts_file = buffer_queue.get()  # Получаем файл из очереди
        if ts_file is None:  # Если обработка завершена
            break
        yield ts_file


def set_invisible():
    return gr.update(visible=False)


def set_visible():
    return gr.update(visible=True)


def set_interactive():
    return gr.update(interactive=True)


def set_disable():
    return gr.update(interactive=False)


# Gradio интерфейс
with gr.Blocks(theme="shivi/calm_seafoam") as fsva:
    intro_text = """
    # Добро пожаловать в сервис анализа выступлений по фигурному катанию!

    Здесь вы можете:
    - Загрузить видео своего выступления.
    - Провести анализ с помощью нейросетевых технологий: обнаружить прыжки, построить скелетную модель спортсмена и автоматически обрезать кадры без прыжков.
    - Получить три варианта фрагментов видео: Normal, Dynamic, Elements.
    - Скачать финальное видео в высоком качестве с эффектами, наложением скелета и только ключевыми моментами.

    Вы можете оставить свою электронную почту, чтобы мы отправили ссылку на готовый результат прямо вам. Также вы можете воспользоваться уникальным кодом, который система создаст при загрузке видео. Если вы не хотите указывать почту, запомните этот код, чтобы вернуться за своим результатом позже. Просто введите его в специальном поле, и вы получите доступ к обработанному видео.

    **Начните с загрузки своего видео!**

    #### Загружая видео вы соглашаетесь с Правилами обработки персональных данных и Правилами сервиса.
    """

    gr.Markdown(intro_text)

    with gr.Tab("Загрузить видео"):
        gr.Markdown("### Загрузите ваше видео для анализа")
        video_input = gr.File(label="Загрузите видео для обработки")

        with gr.Row():
            # Переключатель режима отрисовки
            draw_mode = gr.Radio(
                label="Выберите режим отрисовки (в разработке)",
                choices=["Скелет", "Траектория двух точек", "Чистое видео"],
                value="Чистое видео",
                interactive=False,
                visible=False,
            )

            # Переключатель режима отрисовки
            quality_mode = gr.Radio(
                label="Выберите качество (в разработке)",
                choices=["Whatsapp", "Instagram", "Оригинальное качество"],
                value="Оригинальное качество",
                interactive=False,
                visible=False,
            )

        with gr.Row():
            # Кнопка запуска
            run_button = gr.Button(
                "Сформировать короткое видео", interactive=False, visible=False
            )
            # Отключаем кнопку, пока видео не будет загружено
            video_input.change(enable_button, [video_input], run_button)

        # with gr.Row():
        #     Log(log_file, dark=False)

        with gr.Row():
            run_stream_button = gr.Button("Получить видео", interactive=False)
            video_input.change(enable_button, [video_input], run_stream_button)

        with gr.Row():
            video_stream_output = gr.Video(
                label="Анализ...",
                streaming=True,  # Включаем режим стриминга
                width=234,
                height=416,
                show_download_button=False,
                visible=False,
            )

        with gr.Row():
            download_button = gr.Button(
                "Получить все ролики в хорошем качестве",
                visible=False,
            )

        with gr.Row():
            video_outputs = [None, None, None]
            for i in range(3):
                with gr.Column():
                    video_outputs[i] = gr.Video(
                        label="Обработанное видео",
                        width=234,
                        height=416,
                        autoplay=True,
                        loop=True,
                        visible=False,
                    )

        run_stream_button.click(fn=set_visible, outputs=video_stream_output)

        run_stream_button.click(
            process_video,
            inputs=[
                video_input,
                draw_mode,
                quality_mode,
            ],
            outputs=[video_outputs[0], video_outputs[1], video_outputs[2]],
        )

        # Привязка кнопки запуска к стриму
        run_stream_button.click(
            fn=generate_stream_with_buffer,
            inputs=video_input,
            outputs=video_stream_output,
            queue=True,  # Включаем очередь для потокового обновления
        )

        video_outputs[0].change(fn=set_invisible, outputs=run_stream_button)
        video_outputs[0].change(fn=set_visible, outputs=download_button)
        for i in range(3):
            video_outputs[0].change(fn=set_visible, outputs=video_outputs[i])

    with gr.Tab("Поиск готового короткого видео"):
        gr.Markdown(
            "### Введите вашу почту или уникальный код, чтобы найти готовое видео"
        )
        email_input = gr.Textbox(label="Введите почту или уникальный код")
        email_search_button = gr.Button("Искать")

    logger.info("Сервер загружен")

if __name__ == "__main__":
    fsva.launch(server_name="0.0.0.0", server_port=1328)
