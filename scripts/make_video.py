import argparse
import random

import torch
import yaml

from diffusers import DPMSolverMultistepScheduler
from stable_diffusion_videos import StableDiffusionWalkPipeline


def init_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint_id',
                        default="stabilityai/stable-diffusion-2-1",
                        help="checkpoint id on huggingface")
    parser.add_argument('--prompts', nargs='+',
                        help='sequence of prompts')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='seed for each prompt')
    parser.add_argument('--num_interpolation_steps', type=int, nargs='+',
                        help='number of steps between each image')
    parser.add_argument('--output_dir', default="dreams",
                        help='output directory')
    parser.add_argument('--name',
                        help='output sub-directory')
    parser.add_argument('--fps', type=int, default=10,
                        help='frames per second')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='diffusion guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='number of diffusion inference steps')
    parser.add_argument('--height', type=int, default=512,
                        help='output image height')
    parser.add_argument('--width', type=int, default=512,
                        help='output image width')
    parser.add_argument('--upsample', action='store_true',
                        help='upscale x4 using Real-ESRGAN')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--audio_filepath',
                        help='path to audio file')
    parser.add_argument('--audio_offsets', type=int, nargs='+',
                        help='audio offset for each prompt')
    parser.add_argument('--negative_prompt',
                        help='negative prompt (one for all images)')

    parser.add_argument('--cfg',
                        help='yaml config file (overwrites other options)')

    return parser


def parse_args(parser):
    args = parser.parse_args()

    # read config file
    if args.cfg is not None:
        with open(args.cfg) as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
            else:
                raise ValueError(f'bad field in config file: {key}')

    # check for prompts
    if args.prompts is None:
        raise ValueError('no prompt provided')
    if args.seeds is None:
        args.seeds = [random.getrandbits(16) for _ in args.prompts]

    # check audio arguments
    if args.audio_filepath is not None and args.audio_offsets is None:
        raise ValueError('must provide audio_offsets when providing '
                         'audio_filepath')
    if args.audio_offsets is not None and args.audio_filepath is None:
        raise ValueError('must provide audio_filepath when providing '
                         'audio_offsets')

    # check lengths
    if args.audio_offsets is not None:
        if not len(args.prompts) == len(args.seeds) == len(args.audio_offsets):
            raise ValueError('prompts, seeds and audio_offsets must have same '
                             f'length, got lengths {len(args.prompts)}, '
                             f'{len(args.seeds)} and '
                             f'{len(args.audio_offsets)} respectively')
    else:
        if not len(args.prompts) == len(args.seeds):
            raise ValueError('prompts and seeds must have same length, got '
                             f'lengths {len(args.prompts)} and '
                             f'{len(args.seeds)} respectively')

    # set num_interpolation_steps
    if args.audio_offsets is not None \
            and args.num_interpolation_steps is not None:
        raise ValueError('cannot provide both audio_offsets and '
                         'num_interpolation_steps')
    elif args.audio_offsets is not None:
        args.num_interpolation_steps = [
            (b-a)*args.fps for a, b in zip(
                args.audio_offsets, args.audio_offsets[1:]
            )
        ]
    elif args.num_interpolation_steps is not None \
            and not len(args.num_interpolation_steps) == len(args.prompts)-1:
        raise ValueError('num_interpolation_steps must have length '
                         f'len(prompts)-1, got '
                         f'{len(args.num_interpolation_steps)} != '
                         f'{len(args.prompts)-1}')
    else:
        args.num_interpolation_steps = 5

    return args


def main():
    parser = init_arg_parser()
    args = parse_args(parser)

    pipe = StableDiffusionWalkPipeline.from_pretrained(
        args.checkpoint_id,
        torch_dtype=torch.float16,
        revision="fp16",
        feature_extractor=None,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe.walk(
        prompts=args.prompts,
        seeds=args.seeds,
        num_interpolation_steps=args.num_interpolation_steps,
        output_dir=args.output_dir,
        name=args.name,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        upsample=args.upsample,
        batch_size=args.batch_size,
        audio_filepath=args.audio_filepath,
        audio_start_sec=args.audio_offsets,
        negative_prompt=args.negative_prompt,
    )


if __name__ == '__main__':
    main()
