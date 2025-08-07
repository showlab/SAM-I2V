# -*- coding: utf-8 -*-
# @FileName: setup.py
# @Time    : 16/7/25 17:06
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

from setuptools import setup, find_packages

setup(
    name="i2v",
    version="0.1.0",
    description="SAM-I2V: Promptable Video Segmentation",
    author="Haiyang Mei",
    author_email="haiyang.mei@outlook.com",
    url="https://github.com/showlab/SAM-I2V",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "numpy>=2.0.0",
        "opencv-python>=4.10.0",
        "tqdm>=4.66.0",
        "Pillow>=9.4.0",
        "hydra-core>=1.3.2",
        "iopath>=0.1.10",
        "fvcore>=0.1.5",
        "pandas>=2.2.0",
        "scikit-image>=0.24.0",
        "tensorboard>=2.17.0",
        "pycocotools>=2.0.8",
        "tensordict>=0.5.0",
        "submitit>=1.5.0",
        "natsort",
        "jupyter>=1.0.0",
        "matplotlib>=3.9.0",
        "timm",
        "einops",
    ],
    python_requires=">=3.8",
)
