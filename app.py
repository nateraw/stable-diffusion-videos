import gradio as gr
import torch

from stable_diffusion_walk import SCHEDULERS, pipeline, walk


def fn_images(
    prompt,
    seed,
    scheduler,
    guidance_scale,
    num_inference_steps,
    disable_tqdm,
):
    pipeline.set_progress_bar_config(disable=disable_tqdm)
    pipeline.scheduler = SCHEDULERS[scheduler]  # klms, default, ddim
    with torch.autocast("cuda"):
        return pipeline(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=pipeline.device).manual_seed(seed),
        )["sample"][0]


def fn_videos(
    prompt_1,
    seed_1,
    prompt_2,
    seed_2,
    prompt_3,
    seed_3,
    prompt_4,
    seed_4,
    prompt_5,
    seed_5,
    prompt_6,
    seed_6,
    prompt_7,
    seed_7,
    scheduler,
    guidance_scale,
    num_inference_steps,
    num_walk_steps,
    do_loop,
    disable_tqdm,
    use_lerp_for_text,
    name,
):
    prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6, prompt_7]
    seeds = [seed_1, seed_2, seed_3, seed_4, seed_5, seed_6, seed_7]

    prompts = [x for x in prompts if x.strip()]
    seeds = seeds[: len(prompts)]

    video_path = walk(
        do_loop=do_loop,
        make_video=True,
        guidance_scale=guidance_scale,
        prompts=prompts,
        seeds=seeds,
        num_steps=num_walk_steps,
        num_inference_steps=num_inference_steps,
        use_lerp_for_text=use_lerp_for_text,
        name=name,
        scheduler=scheduler,
        disable_tqdm=disable_tqdm,
    )
    return video_path


interface_videos = gr.Interface(
    fn_videos,
    inputs=[
        gr.Textbox("blueberry spaghetti"),
        gr.Slider(0, 1000, 553, step=1),
        gr.Textbox("strawberry spaghetti"),
        gr.Slider(0, 1000, 234, step=1),
        gr.Textbox(""),
        gr.Slider(0, 1000, 42),
        gr.Textbox(""),
        gr.Slider(0, 1000, 42),
        gr.Textbox(""),
        gr.Slider(0, 1000, 42),
        gr.Textbox(""),
        gr.Slider(0, 1000, 42),
        gr.Textbox(""),
        gr.Slider(0, 1000, 42),
        gr.Dropdown(["klms", "ddim", "default"], value="klms"),
        gr.Slider(0.0, 20.0, 8.5),
        gr.Slider(1, 200, 50),
        gr.Slider(3, 240, 10),
        gr.Checkbox(False),
        gr.Checkbox(False),
        gr.Checkbox(False),
        gr.Textbox(
            "stable_diffusion_video",
            placeholder=(
                "Name of this experiment. Change to avoid overwriting previous outputs"
            ),
        ),
    ],
    outputs=gr.Video(),
)

interface_images = gr.Interface(
    fn_images,
    inputs=[
        gr.Textbox("blueberry spaghetti"),
        gr.Slider(0, 1000, 553, step=1),
        gr.Dropdown(["klms", "ddim", "default"], value="klms"),
        gr.Slider(0.0, 20.0, 8.5),
        gr.Slider(1, 200, 50),
        gr.Checkbox(False),
    ],
    outputs=gr.Image(type="pil"),
)

interface = gr.TabbedInterface(
    [interface_images, interface_videos], ["Images!", "Videos!"]
)

if __name__ == "__main__":
    interface.launch(debug=True)
