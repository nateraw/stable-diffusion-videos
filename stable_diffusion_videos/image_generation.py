import json
import math
import random
import time
from pathlib import Path
from uuid import uuid4

import torch
from diffusers import __version__ as diffusers_version
from huggingface_hub import CommitOperationAdd, create_commit, create_repo

from .upsampling import RealESRGANModel
from .utils import pad_along_axis


def get_all_files(root: Path):
    dirs = [root]
    while len(dirs) > 0:
        dir = dirs.pop()
        for candidate in dir.iterdir():
            if candidate.is_file():
                yield candidate
            if candidate.is_dir():
                dirs.append(candidate)


def get_groups_of_n(n: int, iterator):
    assert n > 1
    buffer = []
    for elt in iterator:
        if len(buffer) == n:
            yield buffer
            buffer = []
        buffer.append(elt)
    if len(buffer) != 0:
        yield buffer


def upload_folder_chunked(
    repo_id: str,
    upload_dir: Path,
    n: int = 100,
    private: bool = False,
    create_pr: bool = False,
):
    """Upload a folder to the Hugging Face Hub in chunks of n files at a time.

    Args:
        repo_id (str): The repo id to upload to.
        upload_dir (Path): The directory to upload.
        n (int, *optional*, defaults to 100): The number of files to upload at a time.
        private (bool, *optional*): Whether to upload the repo as private.
        create_pr (bool, *optional*): Whether to create a PR after uploading instead of commiting directly.
    """

    url = create_repo(repo_id, exist_ok=True, private=private, repo_type="dataset")
    print(f"Uploading files to: {url}")

    root = Path(upload_dir)
    if not root.exists():
        raise ValueError(f"Upload directory {root} does not exist.")

    for i, file_paths in enumerate(get_groups_of_n(n, get_all_files(root))):
        print(f"Committing {file_paths}")
        operations = [
            CommitOperationAdd(
                path_in_repo=f"{file_path.parent.name}/{file_path.name}",
                path_or_fileobj=str(file_path),
            )
            for file_path in file_paths
        ]
        create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload part {i}",
            repo_type="dataset",
            create_pr=create_pr,
        )


def generate_input_batches(pipeline, prompts, seeds, batch_size, height, width):
    if len(prompts) != len(seeds):
        raise ValueError("Number of prompts and seeds must be equal.")

    embeds_batch, noise_batch = None, None
    batch_idx = 0
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        embeds = pipeline.embed_text(prompt)
        noise = torch.randn(
            (1, pipeline.unet.in_channels, height // 8, width // 8),
            device=pipeline.device,
            generator=torch.Generator(device="cpu" if pipeline.device.type == "mps" else pipeline.device).manual_seed(
                seed
            ),
        )
        embeds_batch = embeds if embeds_batch is None else torch.cat([embeds_batch, embeds])
        noise_batch = noise if noise_batch is None else torch.cat([noise_batch, noise])
        batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == len(prompts)
        if not batch_is_ready:
            continue
        yield batch_idx, embeds_batch.type(torch.cuda.HalfTensor), noise_batch.type(torch.cuda.HalfTensor)
        batch_idx += 1
        del embeds_batch, noise_batch
        torch.cuda.empty_cache()
        embeds_batch, noise_batch = None, None


def generate_images(
    pipeline,
    prompt,
    batch_size=1,
    num_batches=1,
    seeds=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    output_dir="./images",
    image_file_ext=".jpg",
    upsample=False,
    height=512,
    width=512,
    eta=0.0,
    push_to_hub=False,
    repo_id=None,
    private=False,
    create_pr=False,
    name=None,
):
    """Generate images using the StableDiffusion pipeline.

    Args:
        pipeline (StableDiffusionWalkPipeline): The StableDiffusion pipeline instance.
        prompt (str): The prompt to use for the image generation.
        batch_size (int, *optional*, defaults to 1): The batch size to use for image generation.
        num_batches (int, *optional*, defaults to 1): The number of batches to generate.
        seeds (list[int], *optional*): The seeds to use for the image generation.
        num_inference_steps (int, *optional*, defaults to 50): The number of inference steps to take.
        guidance_scale (float, *optional*, defaults to 7.5): The guidance scale to use for image generation.
        output_dir (str, *optional*, defaults to "./images"): The output directory to save the images to.
        image_file_ext (str, *optional*, defaults to '.jpg'): The image file extension to use.
        upsample (bool, *optional*, defaults to False): Whether to upsample the images.
        height (int, *optional*, defaults to 512): The height of the images to generate.
        width (int, *optional*, defaults to 512): The width of the images to generate.
        eta (float, *optional*, defaults to 0.0): The eta parameter to use for image generation.
        push_to_hub (bool, *optional*, defaults to False): Whether to push the generated images to the Hugging Face Hub.
        repo_id (str, *optional*): The repo id to push the images to.
        private (bool, *optional*): Whether to push the repo as private.
        create_pr (bool, *optional*): Whether to create a PR after pushing instead of commiting directly.
        name (str, *optional*, defaults to current timestamp str): The name of the sub-directory of
            output_dir to save the images to.
    """
    if push_to_hub:
        if repo_id is None:
            raise ValueError("Must provide repo_id if push_to_hub is True.")

    name = name or time.strftime("%Y%m%d-%H%M%S")
    save_path = Path(output_dir) / name
    save_path.mkdir(exist_ok=False, parents=True)
    prompt_config_path = save_path / "prompt_config.json"

    num_images = batch_size * num_batches
    seeds = seeds or [random.choice(list(range(0, 9999999))) for _ in range(num_images)]
    if len(seeds) != num_images:
        raise ValueError("Number of seeds must be equal to batch_size * num_batches.")

    if upsample:
        if getattr(pipeline, "upsampler", None) is None:
            pipeline.upsampler = RealESRGANModel.from_pretrained("nateraw/real-esrgan")
        pipeline.upsampler.to(pipeline.device)

    cfg = dict(
        prompt=prompt,
        guidance_scale=guidance_scale,
        eta=eta,
        num_inference_steps=num_inference_steps,
        upsample=upsample,
        height=height,
        width=width,
        scheduler=dict(pipeline.scheduler.config),
        tiled=pipeline.tiled,
        diffusers_version=diffusers_version,
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
    )
    prompt_config_path.write_text(json.dumps(cfg, indent=2, sort_keys=False))

    frame_index = 0
    frame_filepaths = []
    for batch_idx, embeds, noise in generate_input_batches(
        pipeline, [prompt] * num_images, seeds, batch_size, height, width
    ):
        print(f"Generating batch {batch_idx}")

        outputs = pipeline(
            text_embeddings=embeds,
            latents=noise,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            height=height,
            width=width,
            output_type="pil" if not upsample else "numpy",
        )["images"]
        if upsample:
            images = []
            for output in outputs:
                images.append(pipeline.upsampler(output))
        else:
            images = outputs

        for image in images:
            frame_filepath = save_path / f"{seeds[frame_index]}{image_file_ext}"
            image.save(frame_filepath)
            frame_filepaths.append(str(frame_filepath))
            frame_index += 1

    return frame_filepaths

    if push_to_hub:
        upload_folder_chunked(repo_id, save_path, private=private, create_pr=create_pr)


def generate_images_flax(
    pipeline,
    params,
    prompt,
    batch_size=1,
    num_batches=1,
    seeds=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    output_dir="./images",
    image_file_ext=".jpg",
    upsample=False,
    height=512,
    width=512,
    push_to_hub=False,
    repo_id=None,
    private=False,
    create_pr=False,
    name=None,
):
    import jax
    from flax.training.common_utils import shard

    """Generate images using the StableDiffusion pipeline.

    Args:
        pipeline (StableDiffusionWalkPipeline): The StableDiffusion pipeline instance.
        params (`Union[Dict, FrozenDict]`): The model parameters.
        prompt (str): The prompt to use for the image generation.
        batch_size (int, *optional*, defaults to 1): The batch size to use for image generation.
        num_batches (int, *optional*, defaults to 1): The number of batches to generate.
        seeds (int, *optional*): The seed to use for the image generation.
        num_inference_steps (int, *optional*, defaults to 50): The number of inference steps to take.
        guidance_scale (float, *optional*, defaults to 7.5): The guidance scale to use for image generation.
        output_dir (str, *optional*, defaults to "./images"): The output directory to save the images to.
        image_file_ext (str, *optional*, defaults to '.jpg'): The image file extension to use.
        upsample (bool, *optional*, defaults to False): Whether to upsample the images.
        height (int, *optional*, defaults to 512): The height of the images to generate.
        width (int, *optional*, defaults to 512): The width of the images to generate.
        push_to_hub (bool, *optional*, defaults to False): Whether to push the generated images to the Hugging Face Hub.
        repo_id (str, *optional*): The repo id to push the images to.
        private (bool, *optional*): Whether to push the repo as private.
        create_pr (bool, *optional*): Whether to create a PR after pushing instead of commiting directly.
        name (str, *optional*, defaults to current timestamp str): The name of the sub-directory of
            output_dir to save the images to.
    """
    if push_to_hub:
        if repo_id is None:
            raise ValueError("Must provide repo_id if push_to_hub is True.")

    name = name or time.strftime("%Y%m%d-%H%M%S")
    save_path = Path(output_dir) / name
    save_path.mkdir(exist_ok=False, parents=True)
    prompt_config_path = save_path / "prompt_config.json"

    num_images = batch_size * num_batches
    seeds = seeds or random.choice(list(range(0, 9999999)))
    prng_seed = jax.random.PRNGKey(seeds)

    if upsample:
        if getattr(pipeline, "upsampler", None) is None:
            pipeline.upsampler = RealESRGANModel.from_pretrained("nateraw/real-esrgan")
            if not torch.cuda.is_available():
                print("Upsampling is recommended to be done on a GPU, as it is very slow on CPU")
            else:
                pipeline.upsampler = pipeline.upsampler.cuda()

    cfg = dict(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        upsample=upsample,
        height=height,
        width=width,
        scheduler=dict(pipeline.scheduler.config),
        # tiled=pipeline.tiled,
        diffusers_version=diffusers_version,
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
    )
    prompt_config_path.write_text(json.dumps(cfg, indent=2, sort_keys=False))

    NUM_TPU_CORES = jax.device_count()
    jit = True  # force jit, assume params are already sharded
    batch_size_total = NUM_TPU_CORES * batch_size if jit else batch_size

    def generate_input_batches(prompts, batch_size):
        prompt_batch = None
        for batch_idx in range(math.ceil(len(prompts) / batch_size)):
            prompt_batch = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            yield batch_idx, prompt_batch

    frame_index = 0
    frame_filepaths = []
    for batch_idx, prompt_batch in generate_input_batches([prompt] * num_images, batch_size_total):
        # This batch size correspond to each TPU core, so we are generating batch_size * NUM_TPU_CORES images
        print(f"Generating batches: {batch_idx*NUM_TPU_CORES} - {min((batch_idx+1)*NUM_TPU_CORES, num_batches)}")
        prompt_ids_batch = pipeline.prepare_inputs(prompt_batch)
        prng_seed_batch = prng_seed

        if jit:
            padded = False
            # Check if len of prompt_batch is multiple of NUM_TPU_CORES, if not pad its ids
            if len(prompt_batch) % NUM_TPU_CORES != 0:
                padded = True
                pad_size = NUM_TPU_CORES - (len(prompt_batch) % NUM_TPU_CORES)
                # Pad embeds_batch and noise_batch with zeros in batch dimension
                prompt_ids_batch = pad_along_axis(prompt_ids_batch, pad_size, axis=0)

            prompt_ids_batch = shard(prompt_ids_batch)
            prng_seed_batch = jax.random.split(prng_seed, jax.device_count())

        outputs = pipeline(
            params,
            prng_seed=prng_seed_batch,
            prompt_ids=prompt_ids_batch,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pil" if not upsample else "numpy",
            jit=jit,
        )["images"]

        if jit:
            # check if we padded and remove that padding from outputs
            if padded:
                outputs = outputs[:-pad_size]

        if upsample:
            images = []
            for output in outputs:
                images.append(pipeline.upsampler(output))
        else:
            images = outputs

        for image in images:
            uuid = str(uuid4())
            frame_filepath = save_path / f"{uuid}{image_file_ext}"
            image.save(frame_filepath)
            frame_filepaths.append(str(frame_filepath))
            frame_index += 1

    return frame_filepaths

    if push_to_hub:
        upload_folder_chunked(repo_id, save_path, private=private, create_pr=create_pr)
