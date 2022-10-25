from stable_diffusion_videos import StableDiffusionWalkPipeline, Interface

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

interface = Interface(pipe)

if __name__ == '__main__':
    interface.launch(debug=True)
