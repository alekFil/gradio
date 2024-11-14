import re

import gradio as gr
from gradio_log import Log
from utils.logger import setup_logger
from utils.main_process import process_video

# Инициализируем логгер
log_file = "app/resources/logs/main.log"
logger = setup_logger("main", log_file)
hash_pattern = re.compile(r"/([a-f0-9]{64})/")


def enable_button(video_file):
    # Включаем кнопку только если видео загружено
    is_loaded = bool(video_file)
    if is_loaded:
        logger.debug("Видео загружено, разблокирована кнопка запуска")
        logger.debug(f"Загружено видео: {hash_pattern.search(video_file).group(1)}")
    else:
        logger.debug("Загруженное видео удалено, кнопка запуска заблокирована")
    return gr.update(interactive=is_loaded)


# Gradio интерфейс
with gr.Blocks() as fsva:
    gr.Markdown("## Генерация коротких видео прыжков")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Загрузите видео для обработки",
                # Отключаем конкретный тип файла во избежание
                # автоматической перекодировки
                # format="mp4",
                autoplay=True,
                sources="upload",
                height=416,
            )

        with gr.Column():
            video_output = gr.Video(
                label="Обработанное видео",
                width=234,
                height=416,
                autoplay=True,
                loop=True,
            )

    with gr.Row():
        draw_mode = gr.Radio(
            label="Выберите режим отрисовки (в разработке)",
            choices=["Скелет", "Траектория двух точек", "Чистое видео"],
            value="Чистое видео",
        )

        # Переключатель режима отрисовки
        quality_mode = gr.Radio(
            label="Выберите качество (в разработке)",
            choices=["Whatsapp", "Instagram", "Оригинальное качество"],
            value="Оригинальное качество",
        )

    with gr.Row():
        # Кнопка запуска
        run_button = gr.Button("Сформировать короткое видео", interactive=False)
        # Отключаем кнопку, пока видео не будет загружено
        video_input.change(enable_button, [video_input], run_button)

    with gr.Row():
        Log(log_file, dark=False)
        # Переключатель режима отрисовки

    # Настраиваем кнопку запуска, чтобы она выводила
    # короткое видео в соседний столбец
    run_button.click(
        process_video,
        inputs=[
            video_input,
            draw_mode,
            quality_mode,
        ],
        outputs=video_output,
    )

    logger.info("Сервер загружен")

if __name__ == "__main__":
    fsva.launch(server_name="0.0.0.0", server_port=1328)
