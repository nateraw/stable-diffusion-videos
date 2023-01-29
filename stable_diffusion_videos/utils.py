from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor


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

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
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


def pad_along_axis(array: np.ndarray, pad_size: int, axis: int = 0) -> np.ndarray:
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)
