import gradio as gr
from utils.main_process import process_video


def enable_button(video_file):
    # Включаем кнопку только если видео загружено
    return gr.update(interactive=bool(video_file))


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
            )

            # Переключатель режима отрисовки
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

            # Кнопка запуска
            run_button = gr.Button("Сформировать короткое видео", interactive=False)
            # Отключаем кнопку, пока видео не будет загружено
            video_input.change(enable_button, [video_input], run_button)

        with gr.Column():
            video_output = gr.Video(
                label="Обработанное видео",
                width=360,
                height=640,
                autoplay=True,
                loop=True,
            )

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

if __name__ == "__main__":
    fsva.launch(server_name="0.0.0.0", server_port=1328)
