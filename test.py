import debugpy
import torch
import typer

from stable_diffusion_videos import StableDiffusionWalkPipeline


app = typer.Typer()


@app.command()
def torchop(debug: bool = False):
    if debug:
        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
    ).to("cuda")

    video_path = pipeline.walk(
        ["a cat", "a dog"],
        [42, 1337],
        fps=2,  # use 5 for testing, 25 or 30 for better quality
        num_interpolation_steps=2,  # use 3-5 for testing, 30 or more for better results
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    )
    print(f"Saved video to {video_path}")


@app.command()
def flaxop(debug: bool = False):
    from flax.jax_utils import replicate
    import jax.numpy as jnp
    from stable_diffusion_videos import FlaxStableDiffusionWalkPipeline

    if debug:
        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    pipeline, params = FlaxStableDiffusionWalkPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jnp.bfloat16
    )
    p_params = replicate(params)

    video_path = pipeline.walk(
        p_params,
        ["a cat", "a dog"],
        [42, 1337],
        fps=2,  # use 5 for testing, 25 or 30 for better quality
        num_interpolation_steps=2,  # use 3-5 for testing, 30 or more for better results
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    )
    print(f"Saved video to {video_path}")


@app.command()
def flax_stable():
    import jax
    from flax.jax_utils import replicate
    from diffusers import FlaxStableDiffusionPipeline

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jax.numpy.bfloat16
    )
    # p_params = replicate(params)
    prng_seed = jax.random.PRNGKey(42)
    prompt = "a dog with a cat"
    num_samples = 2
    # prompt = num_samples * [prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)
    print(prompt_ids.shape)
    images = pipeline(prompt_ids, params, prng_seed, 5, jit=False).images
    print(images.shape)


if __name__ == "__main__":
    # run the app
    app()
