from stable_diffusion_videos import StableDiffusionWalkPipeline, Interface

from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import torch


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

interface = Interface(pipe)

if __name__ == '__main__':
    interface.launch(debug=True)
