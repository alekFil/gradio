import os

import gradio as gr


def process_video(
    video_file,
    padding,
    draw_mode,
    step,
    model_choice,
    progress=gr.Progress(track_tqdm=True),
):
    return os.path.join("app/output/processed_video_compatible.mp4")
