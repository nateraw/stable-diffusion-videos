# Experimental app to help with the process of generating music videos
# Requires youtube-dl to be installed
# pip install youtube-dl

import gradio as gr
import librosa
from pathlib import Path
import numpy as np
import random
from io import BytesIO
import soundfile as sf
from matplotlib import pyplot as plt

from stable_diffusion_videos import StableDiffusionWalkPipeline, generate_images, get_timesteps_arr

from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import torch
import youtube_dl
import os

pipe = StableDiffusionWalkPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None,
    vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda"),
    scheduler=LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
).to("cuda")

if is_xformers_available():
    pipe.enable_xformers_memory_efficient_attention()

def download_example_clip(url, output_dir='./', output_filename='%(title)s.%(ext)s'):
    if (Path(output_dir) / output_filename).exists():
        return str(Path(output_dir) / output_filename)

    files_before = os.listdir(output_dir) if os.path.exists(output_dir) else []
    ydl_opts = {
        'outtmpl': str(Path(output_dir) / output_filename),
        'format': 'bestaudio',
        'extract-audio': True,
        'audio-format': 'mp3',
        'audio-quality': 0,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    files_after = os.listdir(output_dir)
    return str(Path(output_dir) / list(set(files_after) - set(files_before))[0])
    
def audio_data_to_buffer(y, sr):
    audio_filepath = BytesIO()
    audio_filepath.name = 'audio.wav'
    sf.write(audio_filepath, y, samplerate=sr, format='WAV')
    audio_filepath.seek(0)
    return audio_filepath


def plot_array(y):
    fig = plt.figure()
    x = np.arange(y.shape[0]) 
    plt.title("Line graph") 
    plt.xlabel("X axis") 
    plt.ylabel("Y axis") 
    plt.plot(x, y, color ="red") 
    plt.savefig('timesteps_chart.png')
    return fig

def on_slice_btn_click(audio, audio_start_sec, duration, fps, smooth, margin):
    if audio is None:
        return [
            gr.update(visible=False),
            gr.update(visible=False),
        ]

    y, sr = librosa.load(audio, offset=audio_start_sec, duration=duration)
    T = get_timesteps_arr(
        audio_data_to_buffer(y, sr),
        0,
        duration,
        fps=fps,
        margin=margin,
        smooth=smooth,
    )
    return [gr.update(value=(sr, y), visible=True), gr.update(value=plot_array(T), visible=True)]

def on_audio_change_or_clear(audio):
    if audio is None:
        return [
            gr.update(visible=False),
            gr.update(visible=False)
        ]
    
    duration = librosa.get_duration(filename=audio)
    return [
        gr.update(maximum=int(duration), visible=True),
        gr.update(maximum=int(min(10, duration)), visible=True)
    ]

def on_update_weight_settings_btn_click(sliced_audio, duration, fps, smooth, margin):
    if sliced_audio is None:
        return gr.update(visible=False)

    T = get_timesteps_arr(
        sliced_audio,
        0,
        duration,
        fps=fps,
        margin=margin,
        smooth=smooth,
    )
    return gr.update(value=plot_array(T), visible=True)


def on_generate_images_btn_click(
    prompt_a,
    prompt_b,
    seed_a,
    seed_b,
    output_dir,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    upsample,
):
    output_dir = Path(output_dir) / 'images'

    if seed_a == -1:
        seed_a = random.randint(0, 9999999)
    if seed_b == -1:
        seed_b = random.randint(0, 9999999)

    image_a_fpath = generate_images(
        pipe,
        prompt_a,
        seeds=[seed_a],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        upsample=upsample,
        output_dir=output_dir
    )[0]
    image_b_fpath = generate_images(
        pipe,
        prompt_b,
        seeds=[seed_b],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        upsample=upsample,
        output_dir=output_dir
    )[0]

    return [
        gr.update(value=image_a_fpath, visible=True),
        gr.update(value=image_b_fpath, visible=True),
        gr.update(value=seed_a),
        gr.update(value=seed_b),
    ]

def on_generate_music_video_btn_click(
    audio_filepath,
    audio_start_sec,
    duration,
    fps,
    smooth,
    margin,
    prompt_a,
    prompt_b,
    seed_a,
    seed_b,
    batch_size,
    output_dir,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    upsample,
):

    if audio_filepath is None:
        return gr.update(visible=False)

    video_filepath = pipe.walk(
        prompts=[prompt_a, prompt_b],
        seeds=[seed_a, seed_b],
        num_interpolation_steps=int(duration * fps),
        output_dir=output_dir,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        upsample=upsample,
        batch_size=batch_size,
        audio_filepath=audio_filepath,
        audio_start_sec=audio_start_sec,
        margin=margin,
        smooth=smooth,
    )
    return gr.update(value=video_filepath, visible=True)


audio_start_sec = gr.Slider(0, 10, 0, step=1, label="Start (sec)", interactive=True)
duration = gr.Slider(0, 10, 1, step=1, label="Duration (sec)", interactive=True)
slice_btn = gr.Button("Slice Audio")

sliced_audio = gr.Audio(type='filepath')
wav_plot = gr.Plot(label="Interpolation Weights Per Frame")

fps = gr.Slider(1, 60, 12, step=1, label="FPS", interactive=True)
smooth = gr.Slider(0, 1, 0.0, label="Smoothing", interactive=True)
margin = gr.Slider(1.0, 20.0, 1.0, step=0.5, label="Margin Max", interactive=True)
update_weight_settings_btn = gr.Button("Update Interpolation Weights")

prompt_a = gr.Textbox(value='blueberry spaghetti', label="Prompt A")
prompt_b = gr.Textbox(value='strawberry spaghetti', label="Prompt B")
seed_a = gr.Number(-1, label="Seed A", precision=0, interactive=True)
seed_b = gr.Number(-1, label="Seed B", precision=0, interactive=True)
generate_images_btn = gr.Button("Generate Images")
image_a = gr.Image(visible=False, label="Image A")
image_b = gr.Image(visible=False, label="Image B")

batch_size = gr.Slider(1, 32, 1, step=1, label="Batch Size", interactive=True)
generate_music_video_btn = gr.Button("Generate Music Video")
video = gr.Video(visible=False, label="Video")

STEP_1_MARKDOWN = """
## 1. Upload Some Audio

Upload an audio file to use as the source for the music video.
"""

STEP_2_MARKDOWN = """
## 2. Slice Portion of Audio for Generated Clip

Here you can slice a portion of the audio to use for the generated music video. The longer the audio, the more frames will be generated (which will take longer).

I suggest you use this app to make music videos in segments of 5-10 seconds at a time. Then, you can stitch the videos together using a video editor or ffmpeg later.

**Warning**: If your audio file is short, I do no check that the duration you chose is not longer than the audio. It may cause some issues, so just be mindful of that.
"""

STEP_3_MARKDOWN = """
## 3. Set Interpolation Weight Settings

This section lets you play with the settings used to configure how we move through the latent space given the audio you sliced.

If you look at the graph on the right, you'll see in the X-axis how many frames. The Y-axis is the weight of Image A as we move through the latent space.

If you listen to the audio slice and look at the graph, you should see bumps at points where the audio energy is high (in our case, percussive energy).
"""

STEP_4_MARKDOWN = """
## 4. Select Prompts, Seeds, Settings, and Generate Images

Here you can select the settings for image generation.

Then, you can select prompts and seeds for generating images.

  - Image A will be first frame of the generated video.
  - Image B will be last frame of the generated video.
  - The video will be generated by interpolating between the two images using the audio you provided.

If you set the seeds to -1, a random seed will be used and saved for you, so you can explore different images given the same prompt.
"""


with gr.Blocks() as demo:
    gr.Markdown(STEP_1_MARKDOWN)
    audio = gr.Audio(type='filepath', interactive=True)
    gr.Examples(
        [
            download_example_clip(
                url='https://soundcloud.com/nateraw/thoughts',
                output_dir='./music',
                output_filename='thoughts.mp3'
            )
        ],
        inputs=audio,
        outputs=[audio_start_sec, duration],
        fn=on_audio_change_or_clear,
        cache_examples=False
    )
    audio.change(on_audio_change_or_clear, audio, [audio_start_sec, duration])
    audio.clear(on_audio_change_or_clear, audio, [audio_start_sec, duration])

    gr.Markdown(STEP_2_MARKDOWN)
    audio_start_sec.render()
    duration.render()
    slice_btn.render()

    slice_btn.click(on_slice_btn_click, [audio, audio_start_sec, duration, fps, smooth, margin], [sliced_audio, wav_plot])
    sliced_audio.render()

    gr.Markdown(STEP_3_MARKDOWN)

    with gr.Row():
        with gr.Column(scale=4):
            fps.render()
            smooth.render()
            margin.render()
            update_weight_settings_btn.render()
            update_weight_settings_btn.click(
                on_update_weight_settings_btn_click,
                [sliced_audio, duration, fps, smooth, margin],
                wav_plot
            )
        with gr.Column(scale=3):
            wav_plot.render()

    gr.Markdown(STEP_4_MARKDOWN)
    
    with gr.Accordion("Additional Settings", open=False):
        output_dir = gr.Textbox(value='./dreams', label="Output Directory")
        num_inference_steps = gr.Slider(1, 200, 50, step=10, label="Diffusion Inference Steps", interactive=True)
        guidance_scale = gr.Slider(1.0, 25.0, 7.5, step=0.5, label="Guidance Scale", interactive=True)
        height = gr.Slider(512, 1024, 512, step=64, label="Height", interactive=True)
        width = gr.Slider(512, 1024, 512, step=64, label="Width", interactive=True)
        upsample = gr.Checkbox(value=False, label="Upsample with Real-ESRGAN")

    with gr.Row():
        with gr.Column(scale=4):
            prompt_a.render()
        with gr.Column(scale=1):
            seed_a.render()

    with gr.Row():
        with gr.Column(scale=4):
            prompt_b.render()
        with gr.Column(scale=1):
            seed_b.render()

    generate_images_btn.render()

    with gr.Row():
        with gr.Column(scale=1):
            image_a.render()
        with gr.Column(scale=1):
            image_b.render()

    generate_images_btn.click(
        on_generate_images_btn_click,
        [prompt_a, prompt_b, seed_a, seed_b, output_dir, num_inference_steps, guidance_scale, height, width, upsample],
        [image_a, image_b, seed_a, seed_b]
    )

    gr.Markdown("## 5. Generate Music Video")
    # TODO - add equivalent code snippet to generate music video
    batch_size.render()
    generate_music_video_btn.render()
    generate_music_video_btn.click(
        on_generate_music_video_btn_click,
        [audio, audio_start_sec, duration, fps, smooth, margin, prompt_a, prompt_b, seed_a, seed_b, batch_size, output_dir, num_inference_steps, guidance_scale, height, width, upsample],
        video
    )
    video.render()


if __name__ == '__main__':
    demo.launch(debug=True)
