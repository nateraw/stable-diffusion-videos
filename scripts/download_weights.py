#!/usr/bin/env python

import os
import shutil
import sys

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import MODEL_CACHE, MODEL_ID, MODEL_VAE

if os.path.exists(MODEL_CACHE):
    shutil.rmtree(MODEL_CACHE)
os.makedirs(MODEL_CACHE, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE,
    revision="fp16",
)

vae = AutoencoderKL.from_pretrained(
    MODEL_VAE,
    cache_dir=MODEL_CACHE,
)