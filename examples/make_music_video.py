from stable_diffusion_videos import StableDiffusionWalkPipeline

from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler
import torch


pipe = StableDiffusionWalkPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    vae=AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema"),
    torch_dtype=torch.float16,
    revision="fp16",
    safety_checker=None,
    scheduler=LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
).to("cuda")


# I give you permission to scrape this song :)
# youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 -o "music/thoughts.%(ext)s" https://soundcloud.com/nateraw/thoughts
audio_filepath = 'music/thoughts.mp3'

# Seconds in the song. Here we slice the audio from 0:07-0:16
# Should be same length as prompts/seeds.
audio_offsets = [7, 10, 13, 16]

# Output video frames per second.
# Use lower values for testing (5 or 10), higher values for better quality (30 or 60)
fps = 25

# Convert seconds to frames
# This array should be `len(prompts) - 1` as its steps between prompts.
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

prompts = [
    'Baroque oil painting anime key visual concept art of wanderer above the sea of fog 1 8 1 8 with anime maid, brutalist, dark fantasy, rule of thirds golden ratio, fake detail, trending pixiv fanbox, acrylic palette knife, style of makoto shinkai studio ghibli genshin impact jamie wyeth james gilleard greg rutkowski chiho aoshima',
    'the conscious mind entering the dark wood window into the surreal subconscious dream mind, majestic, dreamlike, surrealist, trending on artstation, by gustavo dore ',
    'Chinese :: by martine johanna and simon stålenhag and chie yoshii and casey weldon and wlop :: ornate, dynamic, particulate, rich colors, intricate, elegant, highly detailed, centered, artstation, smooth, sharp focus, octane render, 3d',
    'Chinese :: by martine johanna and simon stålenhag and chie yoshii and casey weldon and wlop :: ornate, dynamic, particulate, rich colors, intricate, elegant, highly detailed, centered, artstation, smooth, sharp focus, octane render, 3d',
]
seeds = [
    6954010,
    8092009,
    1326004,
    5019608,
]
pipe.walk(
    prompts=prompts,
    seeds=seeds,
    num_interpolation_steps=num_interpolation_steps,
    fps=fps,
    audio_filepath=audio_filepath,
    audio_start_sec=audio_offsets[0],
    batch_size=16,
    num_inference_steps=50,
    guidance_scale=15,
    margin=1.0,
    smooth=0.2,
)
