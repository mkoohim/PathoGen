"""
PathoGen - Histopathology Image Inpainting with Diffusion Models

Setup script for pip installation.
"""
from setuptools import setup, find_packages
import os


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    if not os.path.exists(filename):
        return []
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        # Filter out comments and empty lines
        return [line for line in lines if line and not line.startswith("#")]


# Read the README for long description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()


# Parse requirements
requirements = parse_requirements("requirements.txt")


setup(
    name="pathogen",
    version="1.0.0",
    author="PathoGen Team",
    description="Histopathology Image Inpainting with Diffusion Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PathoGen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
)

