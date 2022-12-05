from setuptools import find_packages, setup


def get_version() -> str:
    rel_path = "stable_diffusion_videos/__init__.py"
    with open(rel_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="stable_diffusion_videos",
    version=get_version(),
    author="Nathan Raw",
    author_email="naterawdata@gmail.com",
    description=(
        "Create 🔥 videos with Stable Diffusion by exploring the latent space and morphing between text prompts."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    install_requires=requirements,
    packages=find_packages(),
)
