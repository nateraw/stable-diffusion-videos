from pathlib import Path

import gradio as gr

from stable_diffusion_videos import generate_images


class Interface:
    def __init__(self, pipeline, params=None):
        self.pipeline = pipeline
        self.params = params  # params in case we are using Flax pipeline
        self.interface_images = gr.Interface(
            self.fn_images,
            inputs=[
                gr.Textbox("blueberry spaghetti", label="Prompt"),
                gr.Slider(1, 24, 1, step=1, label="Batch size"),
                gr.Slider(1, 16, 1, step=1, label="# Batches"),
                gr.Slider(10, 100, 50, step=1, label="# Inference Steps"),
                gr.Slider(5.0, 15.0, 7.5, step=0.5, label="Guidance Scale"),
                gr.Slider(512, 1024, 512, step=64, label="Height"),
                gr.Slider(512, 1024, 512, step=64, label="Width"),
                gr.Checkbox(False, label="Upsample"),
                gr.Textbox("./images", label="Output directory to save results to"),
                # gr.Checkbox(False, label='Push results to Hugging Face Hub'),
                # gr.Textbox("", label='Hugging Face Repo ID to push images to'),
            ],
            outputs=gr.Gallery() if self.params is None else gr.Textbox(),
        )

        self.interface_videos = gr.Interface(
            self.fn_videos,
            inputs=[
                gr.Textbox(
                    "blueberry spaghetti\nstrawberry spaghetti",
                    lines=2,
                    label="Prompts, separated by new line",
                ),
                gr.Textbox("42\n1337", lines=2, label="Seeds, separated by new line"),
                gr.Slider(
                    3, 1000, 5, step=1, label="# Interpolation Steps between prompts"
                ),
                gr.Slider(3, 60, 5, step=1, label="Output Video FPS"),
                gr.Slider(1, 24, 1, step=1, label="Batch size"),
                gr.Slider(10, 100, 50, step=1, label="# Inference Steps"),
                gr.Slider(5.0, 15.0, 7.5, step=0.5, label="Guidance Scale"),
                gr.Slider(512, 1024, 512, step=64, label="Height"),
                gr.Slider(512, 1024, 512, step=64, label="Width"),
                gr.Checkbox(False, label="Upsample"),
                gr.Textbox("./dreams", label="Output directory to save results to"),
            ],
            outputs=gr.Video(),
        )
        self.interface = gr.TabbedInterface(
            [self.interface_images, self.interface_videos],
            ["Images!", "Videos!"],
        )

    def fn_videos(
        self,
        prompts,
        seeds,
        num_interpolation_steps,
        fps,
        batch_size,
        num_inference_steps,
        guidance_scale,
        height,
        width,
        upsample,
        output_dir,
    ):
        prompts = [x.strip() for x in prompts.split("\n") if x.strip()]
        seeds = [int(x.strip()) for x in seeds.split("\n") if x.strip()]

        kwargs = dict(
            prompts=prompts,
            seeds=seeds,
            num_interpolation_steps=num_interpolation_steps,
            fps=fps,
            height=height,
            width=width,
            output_dir=output_dir,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            upsample=upsample,
            batch_size=batch_size,
        )
        if self.params is not None:
            # Assume Flax pipeline, force jit, params should be already replicated
            kwargs.update(dict(params=self.params, jit=True))
        return self.pipeline.walk(**kwargs)

    def fn_images(
        self,
        prompt,
        batch_size,
        num_batches,
        num_inference_steps,
        guidance_scale,
        height,
        width,
        upsample,
        output_dir,
        repo_id=None,
        push_to_hub=False,
    ):
        if self.params is not None:
            return "Single image generation is not supported for Flax yet. Go to the videos tab."
        image_filepaths = generate_images(
            self.pipeline,
            prompt,
            batch_size=batch_size,
            num_batches=num_batches,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_dir=output_dir,
            image_file_ext=".jpg",
            upsample=upsample,
            height=height,
            width=width,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
            create_pr=False,
        )
        return [(x, Path(x).stem) for x in sorted(image_filepaths)]

    def launch(self, *args, **kwargs):
        self.interface.launch(*args, **kwargs)
