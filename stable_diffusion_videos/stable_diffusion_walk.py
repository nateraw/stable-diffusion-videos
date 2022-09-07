import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)

from .stable_diffusion_pipeline import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    torch_dtype=torch.float16,
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
SCHEDULERS = dict(default=default_scheduler, ddim=ddim_scheduler, klms=klms_scheduler)


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def walk(
    prompts=["blueberry spaghetti", "strawberry spaghetti"],
    seeds=[42, 123],
    num_steps=5,
    output_dir="dreams",
    name="berry_good_spaghetti",
    height=512,
    width=512,
    guidance_scale=7.5,
    eta=0.0,
    num_inference_steps=50,
    do_loop=False,
    make_video=False,
    use_lerp_for_text=False,
    scheduler="klms",  # choices: default, ddim, klms
    disable_tqdm=False,
):
    pipeline.set_progress_bar_config(disable=disable_tqdm)

    pipeline.scheduler = SCHEDULERS[scheduler]

    output_path = Path(output_dir) / name
    output_path.mkdir(exist_ok=True, parents=True)

    assert len(prompts) == len(seeds)

    first_prompt, *prompts = prompts
    embeds_a = pipeline.embed_text(first_prompt)

    first_seed, *seeds = seeds
    latents_a = torch.randn(
        (1, pipeline.unet.in_channels, height // 8, width // 8),
        device=pipeline.device,
        generator=torch.Generator(device=pipeline.device).manual_seed(first_seed),
    )

    if do_loop:
        prompts.append(first_prompt)
        seeds.append(first_seed)

    frame_index = 0
    for prompt, seed in zip(prompts, seeds):
        # Text
        embeds_b = pipeline.embed_text(prompt)

        # Latent Noise
        latents_b = torch.randn(
            (1, pipeline.unet.in_channels, height // 8, width // 8),
            device=pipeline.device,
            generator=torch.Generator(device=pipeline.device).manual_seed(seed),
        )

        for i, t in enumerate(np.linspace(0, 1, num_steps)):
            do_print_progress = (i == 0) or ((frame_index + 1) % 20 == 0)
            if do_print_progress:
                print(f"COUNT: {frame_index+1}/{len(seeds)*num_steps}")

            if use_lerp_for_text:
                embeds = torch.lerp(embeds_a, embeds_b, float(t))
            else:
                embeds = slerp(float(t), embeds_a, embeds_b)
            latents = slerp(float(t), latents_a, latents_b)

            with torch.autocast("cuda"):
                im = pipeline(
                    latents=latents,
                    text_embeddings=embeds,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_inference_steps=num_inference_steps,
                )["sample"][0]

            im.save(output_path / ("frame%06d.jpg" % frame_index))
            frame_index += 1

        embeds_a = embeds_b
        latents_a = latents_b

    if make_video:
        frame_ref_path = str(output_path / "frame%06d.jpg")
        video_path = str(output_path / f"{name}.mp4")
        subprocess.call(
            f"ffmpeg -r 30 -i {frame_ref_path} -vcodec libx264 -crf 10 -pix_fmt yuv420p"
            f" {video_path}".split()
        )
        return video_path


if __name__ == "__main__":
    import fire

    fire.Fire(walk)
