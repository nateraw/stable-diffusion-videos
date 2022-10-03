import json
import subprocess
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn
import numpy as np
import torch
from diffusers.schedulers import (DDIMScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers import ModelMixin

from .stable_diffusion_pipeline import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=True,
    torch_dtype=torch.float16,
    revision="fp16",
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


def make_video_ffmpeg(frame_dir, output_file_name='output.mp4', frame_filename="frame%06d.png", fps=30):
    frame_ref_path = str(frame_dir / frame_filename)
    video_path = str(frame_dir / output_file_name)
    subprocess.call(
        f"ffmpeg -r {fps} -i {frame_ref_path} -vcodec libx264 -crf 10 -pix_fmt yuv420p"
        f" {video_path}".split()
    )
    return video_path


def walk(
    prompts: List[str] = ["blueberry spaghetti", "strawberry spaghetti"],
    seeds: List[int] = [42, 123],
    num_interpolation_steps: Union[int, List[int]] = 5,
    output_dir: str = "dreams",
    name: str = "berry_good_spaghetti",
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    num_inference_steps: int = 50,
    do_loop: bool = False,
    make_video: bool = False,
    use_lerp_for_text: bool = False,
    scheduler: str = "klms",  # choices: default, ddim, klms
    disable_tqdm: bool = False,
    upsample: bool = False,
    fps: int = 30,
    less_vram: bool = False,
    resume: bool = False,
    batch_size: int = 1,
    frame_filename_ext: str = '.png',
    num_steps: Optional[int] = None
):
    """Generate video frames/a video given a list of prompts and seeds.

    Args:
        prompts (List[str], optional): List of . Defaults to ["blueberry spaghetti", "strawberry spaghetti"].
        seeds (List[int], optional): List of random seeds corresponding to given prompts.
        num_interpolation_steps (Union[int, List[int]], optional): Number of steps to walk during each interpolation step. If int is provided, use same number of steps between each prompt. If a list is provided, the size of `num_interpolation_steps` should be `len(prompts) - 1`. Increase values to 60-200 for good results. Defaults to 5.
        output_dir (str, optional): Root dir where images will be saved. Defaults to "dreams".
        name (str, optional): Sub directory of output_dir to save this run's files. Defaults to "berry_good_spaghetti".
        height (int, optional): Height of image to generate. Defaults to 512.
        width (int, optional): Width of image to generate. Defaults to 512.
        guidance_scale (float, optional): Higher = more adherance to prompt. Lower = let model take the wheel. Defaults to 7.5.
        eta (float, optional): ETA. Defaults to 0.0.
        num_inference_steps (int, optional): Number of diffusion steps. Defaults to 50.
        do_loop (bool, optional): Whether to loop from last prompt back to first. Defaults to False.
        make_video (bool, optional): Whether to make a video or just save the images. Defaults to False.
        use_lerp_for_text (bool, optional): Use LERP instead of SLERP for text embeddings when walking. Defaults to True.
        scheduler (str, optional): Which scheduler to use. Defaults to "klms". Choices are "default", "ddim", "klms".
        disable_tqdm (bool, optional): Whether to turn off the tqdm progress bars. Defaults to False.
        upsample (bool, optional): If True, uses Real-ESRGAN to upsample images 4x. Requires it to be installed
            which you can do by running: `pip install git+https://github.com/xinntao/Real-ESRGAN.git`. Defaults to False.
        fps (int, optional): The frames per second (fps) that you want the video to use. Does nothing if make_video is False. Defaults to 30.
        less_vram (bool, optional): Allow higher resolution output on smaller GPUs. Yields same result at the expense of 10% speed. Defaults to False.
        resume (bool, optional): When set to True, resume from provided '<output_dir>/<name>' path. Useful if your run was terminated
            part of the way through.
        batch_size (int, optional): Number of examples per batch fed to pipeline. Increase this until you
            run out of VRAM. Defaults to 1.
        frame_filename_ext (str, optional): File extension to use when saving/resuming. Update this to
            ".jpg" to save or resume generating jpg images instead. Defaults to ".png".
        num_steps(int, optional): **Deprecated** Number of interpolation steps. Please use `num_interpolation_steps` instead.

    Returns:
        str: Path to video file saved if make_video=True, else None.
    """

    if num_steps:
        warn(
            (
                "The `num_steps` kwarg of the `stable_diffusion_videos.walk` fn is deprecated in 0.4.0 and will be removed in 0.5.0. "
                "Please use `num_interpolation_steps` instead. Setting provided num_interpolation_steps to provided num_steps for now."
            ),
            DeprecationWarning,
            stacklevel=2
        )
        num_interpolation_steps = num_steps

    if upsample:
        from .upsampling import PipelineRealESRGAN

        upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')

    if less_vram:
        pipeline.enable_attention_slicing()

    output_path = Path(output_dir) / name
    output_path.mkdir(exist_ok=True, parents=True)
    prompt_config_path = output_path / 'prompt_config.json'

    if not resume:
        # Write prompt info to file in output dir so we can keep track of what we did
        prompt_config_path.write_text(
            json.dumps(
                dict(
                    prompts=prompts,
                    seeds=seeds,
                    num_interpolation_steps=num_interpolation_steps,
                    name=name,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_inference_steps=num_inference_steps,
                    do_loop=do_loop,
                    make_video=make_video,
                    use_lerp_for_text=use_lerp_for_text,
                    scheduler=scheduler,
                    upsample=upsample,
                    fps=fps,
                    height=height,
                    width=width,
                ),
                indent=2,
                sort_keys=False,
            )
        )
    else:
        # When resuming, we load all available info from existing prompt config, using kwargs passed in where necessary
        if not prompt_config_path.exists():
            raise FileNotFoundError(f"You specified resume=True, but no prompt config file was found at {prompt_config_path}")

        data = json.load(open(prompt_config_path))
        prompts = data['prompts']
        seeds = data['seeds']
        # NOTE - num_steps was renamed to num_interpolation_steps. Including it here for backwards compatibility.
        num_interpolation_steps = data.get('num_interpolation_steps') or data.get('num_steps')
        height = data['height'] if 'height' in data else height
        width = data['width'] if 'width' in data else width
        guidance_scale = data['guidance_scale']
        eta = data['eta']
        num_inference_steps = data['num_inference_steps']
        do_loop = data['do_loop']
        make_video = data['make_video']
        use_lerp_for_text = data['use_lerp_for_text']
        scheduler = data['scheduler']
        disable_tqdm=disable_tqdm
        upsample = data['upsample'] if 'upsample' in data else upsample
        fps = data['fps'] if 'fps' in data else fps

        resume_step = int(sorted(output_path.glob(f"frame*{frame_filename_ext}"))[-1].stem[5:])
        print(f"\nResuming {output_path} from step {resume_step}...")


    if upsample:
        from .upsampling import PipelineRealESRGAN

        upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')

    pipeline.set_progress_bar_config(disable=disable_tqdm)
    pipeline.scheduler = SCHEDULERS[scheduler]

    if isinstance(num_interpolation_steps, int):
        num_interpolation_steps = [num_interpolation_steps] * (len(prompts)-1)

    assert len(prompts) == len(seeds) == len(num_interpolation_steps) +1

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
        num_interpolation_steps.append(num_interpolation_steps[0])


    frame_index = 0
    total_frame_count = sum(num_interpolation_steps)
    for prompt, seed, num_step in zip(prompts, seeds, num_interpolation_steps):
        # Text
        embeds_b = pipeline.embed_text(prompt)

        # Latent Noise
        latents_b = torch.randn(
            (1, pipeline.unet.in_channels, height // 8, width // 8),
            device=pipeline.device,
            generator=torch.Generator(device=pipeline.device).manual_seed(seed),
        )

        latents_batch, embeds_batch = None, None
        for i, t in enumerate(np.linspace(0, 1, num_step)):

            frame_filepath = output_path / (f"frame%06d{frame_filename_ext}" % frame_index)
            if resume and frame_filepath.is_file():
                frame_index += 1
                continue

            if use_lerp_for_text:
                embeds = torch.lerp(embeds_a, embeds_b, float(t))
            else:
                embeds = slerp(float(t), embeds_a, embeds_b)
            latents = slerp(float(t), latents_a, latents_b)

            embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
            latents_batch = latents if latents_batch is None else torch.cat([latents_batch, latents])

            del embeds
            del latents
            torch.cuda.empty_cache()

            batch_is_ready = embeds_batch.shape[0] == batch_size or t == 1.0
            if not batch_is_ready:
                continue

            do_print_progress = (i == 0) or ((frame_index) % 20 == 0)
            if do_print_progress:
                print(f"COUNT: {frame_index}/{total_frame_count}")

            with torch.autocast("cuda"):
                outputs = pipeline(
                    latents=latents_batch,
                    text_embeddings=embeds_batch,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_inference_steps=num_inference_steps,
                    output_type='pil' if not upsample else 'numpy'
                )["sample"]

                del embeds_batch
                del latents_batch
                torch.cuda.empty_cache()
                latents_batch, embeds_batch = None, None

                if upsample:
                    images = []
                    for output in outputs:
                        images.append(upsampling_pipeline(output))
                else:
                    images = outputs
            for image in images:
                frame_filepath = output_path / (f"frame%06d{frame_filename_ext}" % frame_index)
                image.save(frame_filepath)
                frame_index += 1

        embeds_a = embeds_b
        latents_a = latents_b

    if make_video:
        return make_video_ffmpeg(output_path, f"{name}.mp4", fps=fps, frame_filename=f"frame%06d{frame_filename_ext}")


if __name__ == "__main__":
    import fire

    fire.Fire(walk)
