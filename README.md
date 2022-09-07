# stable-diffusion-videos

Try it yourself in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nateraw/stable-diffusion-videos/blob/main/stable_diffusion_videos.ipynb)

**Example** - morphing between "blueberry spaghetti" and "strawberry spaghetti"

https://user-images.githubusercontent.com/32437151/188721341-6f28abf9-699b-46b0-a72e-fa2a624ba0bb.mp4

## How it Works

### The Notebook/App

The above notebook allows you to generate videos by interpolating the latent space of [Stable Diffusion](https://github.com/CompVis/stable-diffusion).

You can either dream up different versions of the same prompt, or morph between different text prompts (with seeds set for each for reproducibility).

The app is built with [Gradio](https://gradio.app/), which allows you to interact with the model in a web app. Here's how I suggest you use it:

1. Use the "Images" tab to generate images you like.
    - Find two images you want to morph between
    - These images should use the same settings (guidance scale, scheduler, height, width)
    - Keep track of the seeds/settings you used so you can reproduce them

2. Generate videos using the "Videos" tab
    - Using the images you found from the step above, provide the prompts/seeds you recorded
    - Set the `num_walk_steps` - for testing you can use a small number like 3 or 5, but to get great results you'll want to use something larger (60-200 steps). 
    - You can (and should) use the `name` input to separate out where the images/videos are saved. (Note that currently ffmpeg will not overwrite if you already made a video with the same name. You'll have to use ffmpeg to create the video yourself if the app fails to do so.)

### The Script

#### Setup

Install the package

```
pip install stable_diffusion_videos
```

Authenticate with Hugging Face

```
huggingface-cli login
```

#### Usage

```python
from stable_diffusion_videos import walk

walk(
    prompts=['a cat', 'a dog'],
    seeds=[42, 1337],
    output_dir='dreams',
    name='animals_test',
    guidance_scale=8.5,
    num_steps=5,  # Change to 60-200 for better results...3-5 for testing
    num_inference_steps=50,
    scheduler='klms',
    disable_tqdm=False,  # Set to True to disable tqdm progress bar
    make_video=True,
    use_lerp_for_text=True,  # Use lerp for text embeddings instead of slerp
    do_loop=False,  # Change to True if you want last prompt to loop back to first prompt
)
```

## Credits

This work built off of [a script](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
) shared by [@karpathy](https://github.com/karpathy). The script was modified to [this gist](https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53), which was then updated/modified to this repo. 

## Contributing

You can file any issues/feature requests [here](https://github.com/nateraw/stable-diffusion-videos/issues)

Enjoy 🤗
