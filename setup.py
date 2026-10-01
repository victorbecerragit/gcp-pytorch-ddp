from setuptools import find_packages, setup

setup(
    name="pytorch_ddp_trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ],
    description="Multi-node PyTorch DDP training on Vertex AI",
)