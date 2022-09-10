from pathlib import Path

import cv2
from PIL import Image
from huggingface_hub import hf_hub_download

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError as e:
    raise ImportError(
        "You tried to import realesrgan without having it installed properly. To install RealESRGAN, run:\n\n"
        "pip install git+https://github.com/xinntao/Real-ESRGAN.git"
    )

class PipelineRealESRGAN:
    def __init__(self, model_path, tile=0, tile_pad=10, pre_pad=0, fp32=False):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32
        )

    def __call__(self, image_path, outscale=4, convert_to_pil=True):
        # TODO - this should work on torch tensors coming out of diffusion pipeline so we can stitch these together better
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            img = image_path
            img = (img * 255).round().astype("uint8")
            img = img[:, :, ::-1]

        image, _ = self.upsampler.enhance(img, outscale=outscale)

        if convert_to_pil:
            image = Image.fromarray(image[:, :, ::-1])

        return image

    @classmethod
    def from_pretrained(cls, model_name_or_path='nateraw/real-esrgan'):
        # reuploaded form official ones mentioned here:
        # https://github.com/xinntao/Real-ESRGAN
        if Path(model_name_or_path).exists():
            file = model_name_or_path
        else:
            file = hf_hub_download(model_name_or_path, 'RealESRGAN_x4plus.pth')
        # stable_diffusion_videos/visualize_detection_example_transformed_2.png
        return cls(file)


    def upsample_imagefolder(self, in_dir, out_dir, suffix='out', outfile_ext='.jpg'):
        in_dir, out_dir = Path(in_dir), Path(out_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"Provided input directory {in_dir} does not exist")

        out_dir.mkdir(exist_ok=True, parents=True)
         
        image_paths = [x for x in in_dir.glob('*') if x.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        for image in image_paths:
            im = self(str(image))
            out_filepath = out_dir / (image.stem + suffix + outfile_ext)
            im.save(out_filepath)
