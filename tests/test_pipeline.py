"""Tests require GPU, so they will not be running on CI (unless someone
wants to figure that out for me).

We'll run these locally before pushing to the repo, or at the very least
before making a release.
"""

from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch
from pathlib import Path
from shutil import rmtree

import pytest


TEST_OUTPUT_ROOT = "test_outputs"
SAMPLES_DIR = Path(__file__).parent / "samples"

@pytest.fixture
def pipeline(scope="session"):
    pipe = StableDiffusionWalkPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
    ).to('cuda')
    return pipe


@pytest.fixture
def run_name(request):
    fn_name = request.node.name.lstrip('test_')
    output_path = Path(TEST_OUTPUT_ROOT) / fn_name
    if output_path.exists():
        rmtree(output_path)
    # We could instead yield here and rm the dir after its written.
    # However, I like being able to view the files locally to see if they look right.
    return fn_name


def test_walk_basic(pipeline, run_name):
    video_path = pipeline.walk(
        ['a cat', 'a dog', 'a horse'],
        seeds=[42, 1337, 2022],
        num_interpolation_steps=[3, 3],
        output_dir=TEST_OUTPUT_ROOT,
        name=run_name,
        fps=3,
    )
    assert Path(video_path).exists(), "Video file was not created"


def test_walk_with_audio(pipeline, run_name):
    fps = 6
    audio_offsets = [2, 4, 5, 8]
    num_interpolation_steps = [(b - a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]
    video_path = pipeline.walk(
        ['a cat', 'a dog', 'a horse', 'a cow'],
        seeds=[42, 1337, 4321, 1234],
        num_interpolation_steps=num_interpolation_steps,
        output_dir=TEST_OUTPUT_ROOT,
        name=run_name,
        fps=fps,
        audio_filepath=str(Path(SAMPLES_DIR) / 'choice.wav'),
        audio_start_sec=audio_offsets[0],
        batch_size=16,
    )
    assert Path(video_path).exists(), "Video file was not created"


def test_walk_with_upsampler(pipeline, run_name):
    video_path = pipeline.walk(
        ['a cat', 'a dog', 'a horse'],
        seeds=[42, 1337, 2022],
        num_interpolation_steps=[3, 3],
        output_dir=TEST_OUTPUT_ROOT,
        name=run_name,
        fps=3,
        upsample=True,
    )
    assert Path(video_path).exists(), "Video file was not created"
