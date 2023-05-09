import os
import shutil
import torch
from diffusers import DDIMScheduler, PNDMScheduler
from cog import BasePredictor, Input, Path
from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler

from stable_diffusion_videos import StableDiffusionWalkPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_VAE = "stabilityai/sd-vae-ft-ema"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        vae = AutoencoderKL.from_pretrained(MODEL_VAE, cache_dir=MODEL_CACHE, local_files_only=True)

        self.pipeline = StableDiffusionWalkPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.float16,
            revision="fp16",
            safety_checker=None,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            scheduler=LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        ).to("cuda")

        default_scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        ddim_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        klms_scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.SCHEDULERS = dict(
            default=default_scheduler, ddim=ddim_scheduler, klms=klms_scheduler
        )

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompts: str = Input(
            description="Input prompts, separate each prompt with '|'.",
            default="a cat | a dog | a horse",
        ),
        seeds: str = Input(
            description="Random seed, separated with '|' to use different seeds for each of the prompt provided above. Leave blank to randomize the seed.",
            default=None,
        ),
        scheduler: str = Input(
            description="Choose the scheduler",
            choices=["default", "ddim", "klms"],
            default="klms",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps for each image generated from the prompt",
            ge=1,
            le=500,
            default=50,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        num_steps: int = Input(
            description="Steps for generating the interpolation video. Recommended to set to 3 or 5 for testing, then up it to 60-200 for better results.",
            default=50,
        ),
        fps: int = Input(
            description="Frame rate for the video.", default=15, ge=5, le=60
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        prompts = [p.strip() for p in prompts.split("|")]
        if seeds is None:
            print("Setting Random seeds.")
            seeds = [int.from_bytes(os.urandom(2), "big") for s in range(len(prompts))]
        else:
            seeds = [s.strip() for s in seeds.split("|")]
            for s in seeds:
                assert s.isdigit(), "Please provide integer seed."
            seeds = [int(s) for s in seeds]

            if len(seeds) > len(prompts):
                seeds = seeds[: len(prompts)]
            else:
                seeds_not_set = len(prompts) - len(seeds)
                print("Setting Random seeds.")
                seeds = seeds + [
                    int.from_bytes(os.urandom(2), "big") for s in range(seeds_not_set)
                ]

        print("Seeds used for prompts are:")
        for prompt, seed in zip(prompts, seeds):
            print(f"{prompt}: {seed}")

        # use the default settings for the demo
        height = 512
        width = 512
        disable_tqdm = False

        self.pipeline.set_progress_bar_config(disable=disable_tqdm)
        self.pipeline.scheduler = self.SCHEDULERS[scheduler]

        outdir = "cog_out/out"
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)



        self.pipeline.walk(
            prompts=prompts,
            seeds=seeds,
            num_interpolation_steps=num_steps,
            output_dir="cog_out",
            name="out",
            fps=fps,
            batch_size=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            margin=1.0,
            smooth=0.2,
            height=height,
            width=width,
        )

        video_path = f"cog_out/out/out.mp4"

        return Path(video_path)
