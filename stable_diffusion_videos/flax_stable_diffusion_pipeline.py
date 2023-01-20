import json
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import librosa
import numpy as np
import torch
from diffusers.models import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from diffusers.pipeline_flax_utils import FlaxDiffusionPipeline
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker_flax import (
    FlaxStableDiffusionSafetyChecker,
)
from diffusers.schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from diffusers.utils import deprecate, logging
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard
from packaging import version
from PIL import Image
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False
NUM_TPU_CORES = jax.device_count()

from .upsampling import RealESRGANModel


def pad_along_axis(array: np.ndarray, pad_size: int, axis: int = 0) -> np.ndarray:
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)


def get_timesteps_arr(audio_filepath, offset, duration, fps=30, margin=1.0, smooth=0.0):
    y, sr = librosa.load(audio_filepath, offset=offset, duration=duration)

    # librosa.stft hardcoded defaults...
    # n_fft defaults to 2048
    # hop length is win_length // 4
    # win_length defaults to n_fft
    D = librosa.stft(y, n_fft=2048, hop_length=2048 // 4, win_length=2048)

    # Extract percussive elements
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)
    y_percussive = librosa.istft(D_percussive, length=len(y))

    # Get normalized melspectrogram
    spec_raw = librosa.feature.melspectrogram(y=y_percussive, sr=sr)
    spec_max = np.amax(spec_raw, axis=0)
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    # Resize cumsum of spec norm to our desired number of interpolation frames
    x_norm = np.linspace(0, spec_norm.shape[-1], spec_norm.shape[-1])
    y_norm = np.cumsum(spec_norm)
    y_norm /= y_norm[-1]
    x_resize = np.linspace(0, y_norm.shape[-1], int(duration * fps))

    T = np.interp(x_resize, x_norm, y_norm)

    # Apply smoothing
    return T * (1 - smooth) + np.linspace(0.0, 1.0, T.shape[0]) * smooth


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    # if not isinstance(v0, np.ndarray):
    #     inputs_are_torch = True
    #     input_device = v0.device
    #     v0 = v0.cpu().numpy()
    #     v1 = v1.cpu().numpy()

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

    # if inputs_are_torch:
    #     v2 = torch.from_numpy(v2).to(input_device)

    return v2


def make_video_pyav(
    frames_or_frame_dir: Union[str, Path, torch.Tensor],
    audio_filepath: Union[str, Path] = None,
    fps: int = 30,
    audio_offset: int = 0,
    audio_duration: int = 2,
    sr: int = 22050,
    output_filepath: Union[str, Path] = "output.mp4",
    glob_pattern: str = "*.png",
):
    """
    TODO - docstring here

    frames_or_frame_dir: (Union[str, Path, torch.Tensor]):
        Either a directory of images, or a tensor of shape (T, C, H, W) in range [0, 255].
    """

    # Torchvision write_video doesn't support pathlib paths
    output_filepath = str(output_filepath)

    if isinstance(frames_or_frame_dir, (str, Path)):
        frames = None
        for img in sorted(Path(frames_or_frame_dir).glob(glob_pattern)):
            frame = pil_to_tensor(Image.open(img)).unsqueeze(0)
            frames = frame if frames is None else torch.cat([frames, frame])
    else:
        frames = frames_or_frame_dir

    # TCHW -> THWC
    frames = frames.permute(0, 2, 3, 1)

    if audio_filepath:
        # Read audio, convert to tensor
        audio, sr = librosa.load(
            audio_filepath,
            sr=sr,
            mono=True,
            offset=audio_offset,
            duration=audio_duration,
        )
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        write_video(
            output_filepath,
            frames,
            fps=fps,
            audio_array=audio_tensor,
            audio_fps=sr,
            audio_codec="aac",
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )
    else:
        write_video(
            output_filepath,
            frames,
            fps=fps,
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )

    return output_filepath


class FlaxStableDiffusionWalkPipeline(FlaxDiffusionPipeline):
    r"""
    Pipeline for generating videos by interpolating  Stable Diffusion's latent space.

    This model inherits from [`FlaxDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`FlaxAutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`FlaxCLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.FlaxCLIPTextModel),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`FlaxUNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`FlaxDDIMScheduler`], [`FlaxLMSDiscreteScheduler`], [`FlaxPNDMScheduler`], or
            [`FlaxDPMSolverMultistepScheduler`].
        safety_checker ([`FlaxStableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: FlaxAutoencoderKL,
        text_encoder: FlaxCLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: FlaxUNet2DConditionModel,
        scheduler: Union[
            FlaxDDIMScheduler,
            FlaxPNDMScheduler,
            FlaxLMSDiscreteScheduler,
            FlaxDPMSolverMultistepScheduler,
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        if safety_checker is None:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _get_has_nsfw_concepts(self, features, params):
        has_nsfw_concepts = self.safety_checker(features, params)
        return has_nsfw_concepts

    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # safety_model_params should already be replicated when jit is True
        pil_images = [Image.fromarray(image) for image in images]
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values

        if jit:
            features = shard(features)
            has_nsfw_concepts = _p_get_has_nsfw_concepts(
                self, features, safety_model_params
            )
            has_nsfw_concepts = unshard(has_nsfw_concepts)
            safety_model_params = unreplicate(safety_model_params)
        else:
            has_nsfw_concepts = self._get_has_nsfw_concepts(
                features, safety_model_params
            )

        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()

                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image

            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )

        return images, has_nsfw_concepts

    def _generate(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.PRNGKey,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        latents: Optional[jnp.array] = None,
        neg_prompt_ids: jnp.array = None,
        text_embeddings: Optional[jnp.array] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if text_embeddings is None:
            # get prompt text embeddings
            text_embeddings = self.text_encoder(
                prompt_ids, params=params["text_encoder"]
            )[0]
            # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
            # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
            batch_size = prompt_ids.shape[0]

            max_length = prompt_ids.shape[-1]
        else:
            batch_size = text_embeddings.shape[0]
            # TODO: check if this is enough
            max_length = self.tokenizer.model_max_length

        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="np",
            ).input_ids
        else:
            uncond_input = neg_prompt_ids
        uncond_embeddings = self.text_encoder(
            uncond_input, params=params["text_encoder"]
        )[0]
        context = jnp.concatenate([uncond_embeddings, text_embeddings])

        latents_shape = (
            batch_size,
            self.unet.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = jax.random.normal(
                prng_seed, shape=latents_shape, dtype=jnp.float32
            )
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )

        def loop_body(step, args):
            latents, scheduler_state = args
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            latents_input = self.scheduler.scale_model_input(
                scheduler_state, latents_input, t
            )

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
            ).sample
            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_prediction_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(
                scheduler_state, noise_pred, t, latents
            ).to_tuple()
            return latents, scheduler_state

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"],
            num_inference_steps=num_inference_steps,
            shape=latents.shape,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        if DEBUG:
            # run with python for loop
            for i in range(num_inference_steps):
                latents, scheduler_state = loop_body(i, (latents, scheduler_state))
        else:
            latents, _ = jax.lax.fori_loop(
                0, num_inference_steps, loop_body, (latents, scheduler_state)
            )

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.apply(
            {"params": params["vae"]}, latents, method=self.vae.decode
        ).sample

        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return image

    def __call__(
        self,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.PRNGKey,
        prompt_ids: Optional[jnp.array] = None,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: Union[float, jnp.array] = 7.5,
        latents: jnp.array = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        jit: bool = False,
        neg_prompt_ids: jnp.array = None,
        text_embeddings: Optional[jnp.array] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            params (`Union[Dict, FrozenDict]`): The model parameters.
            prng_seed (`jax.random.PRNGKey`): The random seed used for sampling the noise.
            prompt_ids (`jnp.array`, *optional*, defaults to `None`):
                The prompt or prompts to guide the image generation. If not provided, `text_embeddings` is required.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            latents (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. tensor will ge generated
                by sampling using the supplied random `generator`.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions. NOTE: This argument
                exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a future release.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            neg_prompt_ids (`jnp.array`, *optional*):
                The prompt or prompts ids not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            text_embeddings (`jnp.array`, *optional*, defaults to `None`):
                Pre-generated text embeddings to be used as inputs for image generation. Can be used in place of
                `prompt_ids` to avoid re-computing the embeddings. If not provided, the embeddings will be generated from
                the supplied `prompt`.

        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt_ids is None and text_embeddings is None:
            raise ValueError(
                "Either `prompt_ids` or `text_embeddings` must be provided."
            )

        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                text_embeddings,
            )
        else:
            images = self._generate(
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                latents,
                neg_prompt_ids,
                text_embeddings,
            )

        if self.safety_checker is not None:
            safety_params = params["safety_checker"]
            images_uint8_casted = (images * 255).round().astype("uint8")
            num_devices, batch_size = images.shape[:2]

            images_uint8_casted = np.asarray(images_uint8_casted).reshape(
                num_devices * batch_size, height, width, 3
            )
            images_uint8_casted, has_nsfw_concept = self._run_safety_checker(
                images_uint8_casted, safety_params, jit
            )
            images = np.asarray(images).reshape(
                num_devices * batch_size, height, width, 3
            )

            # block images
            if any(has_nsfw_concept):
                for i, is_nsfw in enumerate(has_nsfw_concept):
                    if is_nsfw:
                        images[i] = np.asarray(images_uint8_casted[i])

            images = images.reshape(num_devices, batch_size, height, width, 3)
        else:
            images = np.asarray(images)
            has_nsfw_concept = False

        if jit:
            images = unshard(images)

        # Convert to PIL
        if output_type == "pil":
            images = self.numpy_to_pil(images)

        if not return_dict:
            return (images, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(
            images=images, nsfw_content_detected=has_nsfw_concept
        )

    def generate_inputs(
        self, params, prompt_a, prompt_b, seed_a, seed_b, noise_shape, T, batch_size
    ):
        embeds_a = self.embed_text(params, prompt_a)
        embeds_b = self.embed_text(params, prompt_b)
        latents_dtype = embeds_a.dtype
        latents_a = self.init_noise(seed_a, noise_shape, latents_dtype)
        latents_b = self.init_noise(seed_b, noise_shape, latents_dtype)

        batch_idx = 0
        embeds_batch, noise_batch = None, None
        for i, t in enumerate(T):
            embeds = slerp(float(t), embeds_a, embeds_b)
            noise = slerp(float(t), latents_a, latents_b)

            embeds_batch = (
                embeds
                if embeds_batch is None
                else np.concatenate([embeds_batch, embeds])
            )
            noise_batch = (
                noise if noise_batch is None else np.concatenate([noise_batch, noise])
            )
            batch_is_ready = embeds_batch.shape[0] == batch_size or i + 1 == T.shape[0]
            if not batch_is_ready:
                continue
            yield batch_idx, embeds_batch, noise_batch
            batch_idx += 1
            del embeds_batch, noise_batch
            # torch.cuda.empty_cache()
            embeds_batch, noise_batch = None, None

    def make_clip_frames(
        self,
        params: Union[Dict, FrozenDict],
        prompt_a: str,
        prompt_b: str,
        seed_a: int,
        seed_b: int,
        num_interpolation_steps: int = 5,
        save_path: Union[str, Path] = "outputs/",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: bool = False,
        batch_size: int = 1,
        image_file_ext: str = ".png",
        T: np.ndarray = None,
        skip: int = 0,
        negative_prompt: str = None,
        jit: bool = False,
    ):
        if negative_prompt is not None:
            raise NotImplementedError(
                "Negative prompt is not supported for make_clip_frames yet."
            )
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        T = T if T is not None else np.linspace(0.0, 1.0, num_interpolation_steps)
        if T.shape[0] != num_interpolation_steps:
            raise ValueError(
                f"Unexpected T shape, got {T.shape}, expected dim 0 to be {num_interpolation_steps}"
            )

        if upsample:
            if getattr(self, "upsampler", None) is None:
                # TODO: port to flax
                self.upsampler = RealESRGANModel.from_pretrained("nateraw/real-esrgan")
                if not torch.cuda.is_available():
                    logger.warning(
                        "Upsampling is recommended to be done on a GPU, as it is very slow on CPU"
                    )
                else:
                    self.upsampler = self.upsampler.cuda()

        seed_a = jax.random.PRNGKey(seed_a)
        seed_b = jax.random.PRNGKey(seed_b)

        text_encoder_params = params["text_encoder"]
        if jit:  # if jit, asume params are replicated
            # for encoding de prompts we run it on a single device
            text_encoder_params = unreplicate(text_encoder_params)

        batch_generator = self.generate_inputs(
            text_encoder_params,
            prompt_a,
            prompt_b,
            seed_a,
            seed_b,
            (1, self.unet.in_channels, height // 8, width // 8),
            T[skip:],
            batch_size=NUM_TPU_CORES * batch_size if jit else batch_size,
        )

        # TODO: convert negative_prompt to neg_prompt_ids

        frame_index = skip
        for _, embeds_batch, noise_batch in batch_generator:
            if jit:
                padded = False
                # Check if embeds_batch 0 dimension is multiple of NUM_TPU_CORES, if not pad
                if embeds_batch.shape[0] % NUM_TPU_CORES != 0:
                    padded = True
                    pad_size = NUM_TPU_CORES - (embeds_batch.shape[0] % NUM_TPU_CORES)
                    # Pad embeds_batch and noise_batch with zeros in batch dimension
                    embeds_batch = pad_along_axis(embeds_batch, pad_size, axis=0)
                    noise_batch = pad_along_axis(noise_batch, pad_size, axis=0)
                embeds_batch = shard(embeds_batch)
                noise_batch = shard(noise_batch)

            outputs = self(
                params,
                prng_seed=None,
                latents=noise_batch,
                text_embeddings=embeds_batch,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil" if not upsample else "numpy",
                neg_prompt_ids=negative_prompt,
                jit=jit,
            )["images"]

            if jit:
                # check if we padded and remove that padding from outputs
                if padded:
                    outputs = outputs[:-pad_size]

            for image in outputs:
                frame_filepath = save_path / (
                    f"frame%06d{image_file_ext}" % frame_index
                )
                # image = image if not upsample else self.upsampler(image)
                image.save(frame_filepath)
                frame_index += 1

    def walk(
        self,
        params: Union[Dict, FrozenDict],
        prompts: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        num_interpolation_steps: Optional[
            Union[int, List[int]]
        ] = 5,  # int or list of int
        output_dir: Optional[str] = "./dreams",
        name: Optional[str] = None,
        image_file_ext: Optional[str] = ".png",
        fps: Optional[int] = 30,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        upsample: Optional[bool] = False,
        batch_size: Optional[int] = 1,
        resume: Optional[bool] = False,
        audio_filepath: str = None,
        audio_start_sec: Optional[Union[int, float]] = None,
        margin: Optional[float] = 1.0,
        smooth: Optional[float] = 0.0,
        negative_prompt: Optional[str] = None,
        jit: bool = False,
    ):
        """Generate a video from a sequence of prompts and seeds. Optionally, add audio to the
        video to interpolate to the intensity of the audio.

        Args:
            prompts (Optional[List[str]], optional):
                list of text prompts. Defaults to None.
            seeds (Optional[List[int]], optional):
                list of random seeds corresponding to prompts. Defaults to None.
            num_interpolation_steps (Union[int, List[int]], *optional*):
                How many interpolation steps between each prompt. Defaults to None.
            output_dir (Optional[str], optional):
                Where to save the video. Defaults to './dreams'.
            name (Optional[str], optional):
                Name of the subdirectory of output_dir. Defaults to None.
            image_file_ext (Optional[str], *optional*, defaults to '.png'):
                The extension to use when writing video frames.
            fps (Optional[int], *optional*, defaults to 30):
                The frames per second in the resulting output videos.
            num_inference_steps (Optional[int], *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (Optional[float], *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (Optional[float], *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            height (Optional[int], *optional*, defaults to None):
                height of the images to generate.
            width (Optional[int], *optional*, defaults to None):
                width of the images to generate.
            upsample (Optional[bool], *optional*, defaults to False):
                When True, upsamples images with realesrgan.
            batch_size (Optional[int], *optional*, defaults to 1):
                Number of images to generate at once.
            resume (Optional[bool], *optional*, defaults to False):
                When True, resumes from the last frame in the output directory based
                on available prompt config. Requires you to provide the `name` argument.
            audio_filepath (str, *optional*, defaults to None):
                Optional path to an audio file to influence the interpolation rate.
            audio_start_sec (Optional[Union[int, float]], *optional*, defaults to 0):
                Global start time of the provided audio_filepath.
            margin (Optional[float], *optional*, defaults to 1.0):
                Margin from librosa hpss to use for audio interpolation.
            smooth (Optional[float], *optional*, defaults to 0.0):
                Smoothness of the audio interpolation. 1.0 means linear interpolation.
            negative_prompt (Optional[str], *optional*, defaults to None):
                Optional negative prompt to use. Same across all prompts.

        This function will create sub directories for each prompt and seed pair.

        For example, if you provide the following prompts and seeds:

        ```
        prompts = ['a dog', 'a cat', 'a bird']
        seeds = [1, 2, 3]
        num_interpolation_steps = 5
        output_dir = 'output_dir'
        name = 'name'
        fps = 5
        ```

        Then the following directories will be created:

        ```
        output_dir
        ├── name
        │   ├── name_000000
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000000.mp4
        │   ├── name_000001
        │   │   ├── frame000000.png
        │   │   ├── ...
        │   │   ├── frame000004.png
        │   │   ├── name_000001.mp4
        │   ├── ...
        │   ├── name.mp4
        |   |── prompt_config.json
        ```

        Returns:
            str: The resulting video filepath. This video includes all sub directories' video clips.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        output_path = Path(output_dir)

        name = name or time.strftime("%Y%m%d-%H%M%S")
        save_path_root = output_path / name
        save_path_root.mkdir(parents=True, exist_ok=True)

        # Where the final video of all the clips combined will be saved
        output_filepath = save_path_root / f"{name}.mp4"

        # If using same number of interpolation steps between, we turn into list
        if not resume and isinstance(num_interpolation_steps, int):
            num_interpolation_steps = [num_interpolation_steps] * (len(prompts) - 1)

        if not resume:
            audio_start_sec = audio_start_sec or 0

        # Save/reload prompt config
        prompt_config_path = save_path_root / "prompt_config.json"
        if not resume:
            prompt_config_path.write_text(
                json.dumps(
                    dict(
                        prompts=prompts,
                        seeds=seeds,
                        num_interpolation_steps=num_interpolation_steps,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        eta=eta,
                        upsample=upsample,
                        height=height,
                        width=width,
                        audio_filepath=audio_filepath,
                        audio_start_sec=audio_start_sec,
                        negative_prompt=negative_prompt,
                    ),
                    indent=2,
                    sort_keys=False,
                )
            )
        else:
            data = json.load(open(prompt_config_path))
            prompts = data["prompts"]
            seeds = data["seeds"]
            num_interpolation_steps = data["num_interpolation_steps"]
            fps = data["fps"]
            num_inference_steps = data["num_inference_steps"]
            guidance_scale = data["guidance_scale"]
            eta = data["eta"]
            upsample = data["upsample"]
            height = data["height"]
            width = data["width"]
            audio_filepath = data["audio_filepath"]
            audio_start_sec = data["audio_start_sec"]
            negative_prompt = data.get("negative_prompt", None)

        for i, (prompt_a, prompt_b, seed_a, seed_b, num_step) in enumerate(
            zip(prompts, prompts[1:], seeds, seeds[1:], num_interpolation_steps)
        ):
            # {name}_000000 / {name}_000001 / ...
            save_path = save_path_root / f"{name}_{i:06d}"

            # Where the individual clips will be saved
            step_output_filepath = save_path / f"{name}_{i:06d}.mp4"

            # Determine if we need to resume from a previous run
            skip = 0
            if resume:
                if step_output_filepath.exists():
                    print(f"Skipping {save_path} because frames already exist")
                    continue

                existing_frames = sorted(save_path.glob(f"*{image_file_ext}"))
                if existing_frames:
                    skip = int(existing_frames[-1].stem[-6:]) + 1
                    if skip + 1 >= num_step:
                        print(f"Skipping {save_path} because frames already exist")
                        continue
                    print(f"Resuming {save_path.name} from frame {skip}")

            audio_offset = audio_start_sec + sum(num_interpolation_steps[:i]) / fps
            audio_duration = num_step / fps

            self.make_clip_frames(
                params,
                prompt_a,
                prompt_b,
                seed_a,
                seed_b,
                num_interpolation_steps=num_step,
                save_path=save_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                height=height,
                width=width,
                upsample=upsample,
                batch_size=batch_size,
                T=get_timesteps_arr(
                    audio_filepath,
                    offset=audio_offset,
                    duration=audio_duration,
                    fps=fps,
                    margin=margin,
                    smooth=smooth,
                )
                if audio_filepath
                else None,
                skip=skip,
                negative_prompt=negative_prompt,
                jit=jit,
            )
            make_video_pyav(
                save_path,
                audio_filepath=audio_filepath,
                fps=fps,
                output_filepath=step_output_filepath,
                glob_pattern=f"*{image_file_ext}",
                audio_offset=audio_offset,
                audio_duration=audio_duration,
                sr=44100,
            )

        return make_video_pyav(
            save_path_root,
            audio_filepath=audio_filepath,
            fps=fps,
            audio_offset=audio_start_sec,
            audio_duration=sum(num_interpolation_steps) / fps,
            output_filepath=output_filepath,
            glob_pattern=f"**/*{image_file_ext}",
            sr=44100,
        )

    def embed_text(
        self, params: Union[Dict, FrozenDict], text: str, negative_prompt=None
    ):
        """Helper to embed some text"""
        prompt_ids = self.prepare_inputs(text)
        embed = self.text_encoder(prompt_ids, params=params)[0]
        return embed

    def init_noise(self, prng_seed, noise_shape, dtype):
        """Helper to initialize noise"""
        noise = jax.random.normal(prng_seed, shape=noise_shape, dtype=dtype)
        return noise

    # TODO: port this behavior to flax
    # @classmethod
    # def from_pretrained(cls, *args, tiled=False, **kwargs):
    #     """Same as diffusers `from_pretrained` but with tiled option, which makes images tilable"""
    #     if tiled:

    #         def patch_conv(**patch):
    #             cls = nn.Conv2d
    #             init = cls.__init__

    #             def __init__(self, *args, **kwargs):
    #                 return init(self, *args, **kwargs, **patch)

    #             cls.__init__ = __init__

    #         patch_conv(padding_mode="circular")

    #     pipeline = super().from_pretrained(*args, **kwargs)
    #     pipeline.tiled = tiled
    #     return pipeline


# Static argnums are pipe, num_inference_steps, height, width. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
# guidance_scale is a scalar, so it's broadcasted to all devices (hence `None`) without needing to be static.
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, None, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 4, 5, 6),
)
def _p_generate(
    pipe,
    prompt_ids,
    params,
    prng_seed,
    num_inference_steps,
    height,
    width,
    guidance_scale,
    latents,
    neg_prompt_ids,
    text_embeddings,
):
    return pipe._generate(
        prompt_ids,
        params,
        prng_seed,
        num_inference_steps,
        height,
        width,
        guidance_scale,
        latents,
        neg_prompt_ids,
        text_embeddings,
    )


@partial(jax.pmap, static_broadcasted_argnums=(0,))
def _p_get_has_nsfw_concepts(pipe, features, params):
    return pipe._get_has_nsfw_concepts(features, params)


def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)
