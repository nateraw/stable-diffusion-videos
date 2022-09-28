import time

import gradio as gr
import torch

from .stable_diffusion_walk import SCHEDULERS, pipeline, walk


def fn_images(
    prompt,
    seed,
    scheduler,
    guidance_scale,
    num_inference_steps,
    disable_tqdm,
    upsample,
):
    if upsample:
        from .upsampling import PipelineRealESRGAN

        upsampling_pipeline = PipelineRealESRGAN.from_pretrained('nateraw/real-esrgan')

    pipeline.set_progress_bar_config(disable=disable_tqdm)
    pipeline.scheduler = SCHEDULERS[scheduler]  # klms, default, ddim
    with torch.autocast("cuda"):
        img = pipeline(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=pipeline.device).manual_seed(seed),
            output_type='pil' if not upsample else 'numpy',
        )["sample"][0]
        return img if not upsample else upsampling_pipeline(img)


def fn_videos(
    prompt_1,
    seed_1,
    prompt_2,
    seed_2,
    scheduler,
    guidance_scale,
    num_inference_steps,
    num_interpolation_steps,
    do_loop,
    disable_tqdm,
    use_lerp_for_text,
    output_dir,
    upsample,
):
    prompts = [prompt_1, prompt_2]
    seeds = [seed_1, seed_2]

    prompts = [x for x in prompts if x.strip()]
    seeds = seeds[: len(prompts)]

    video_path = walk(
        do_loop=do_loop,
        make_video=True,
        guidance_scale=guidance_scale,
        prompts=prompts,
        seeds=seeds,
        num_interpolation_steps=num_interpolation_steps,
        num_inference_steps=num_inference_steps,
        use_lerp_for_text=use_lerp_for_text,
        output_dir=output_dir,
        name=time.strftime("%Y%m%d-%H%M%S"),
        scheduler=scheduler,
        disable_tqdm=disable_tqdm,
        upsample=upsample
    )
    return video_path


interface_videos = gr.Interface(
    fn_videos,
    inputs=[
        gr.Textbox("blueberry spaghetti"),
        gr.Number(42, label='Seed 1', precision=0),
        gr.Textbox("strawberry spaghetti"),
        gr.Number(42, label='Seed 2', precision=0),
        gr.Dropdown(["klms", "ddim", "default"], value="klms"),
        gr.Slider(0.0, 20.0, 8.5),
        gr.Slider(1, 200, 50),
        gr.Slider(3, 240, 10),
        gr.Checkbox(False),
        gr.Checkbox(False),
        gr.Checkbox(True),
        gr.Textbox(
            "dreams",
            placeholder=(
                "Folder where outputs will be saved. Each output will be saved in a new folder."
            ),
        ),
        gr.Checkbox(False),
    ],
    outputs=gr.Video(),
)

interface_images = gr.Interface(
    fn_images,
    inputs=[
        gr.Textbox("blueberry spaghetti"),
        gr.Number(42, label='Seed', precision=0),
        gr.Dropdown(["klms", "ddim", "default"], value="klms"),
        gr.Slider(0.0, 20.0, 8.5),
        gr.Slider(1, 200, 50),
        gr.Checkbox(False),
        gr.Checkbox(False),
    ],
    outputs=gr.Image(type="pil"),
)

interface = gr.TabbedInterface(
    [interface_images, interface_videos], ["Images!", "Videos!"]
)

if __name__ == "__main__":
    interface.launch(debug=True)
