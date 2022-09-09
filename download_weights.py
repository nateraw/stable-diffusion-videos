import os
import sys
import argparse
import torch
from diffusers import StableDiffusionPipeline

os.makedirs("diffusers-cache", exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auth_token",
        required=True,
        help="Authentication token from Huggingface for downloaging weights.",
    )
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        cache_dir="diffusers-cache",
        torch_dtype=torch.float16,
        use_auth_token=args.auth_token,
        revision="fp16"
    )