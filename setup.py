from setuptools import find_packages, setup

setup(
    name="precision-oncology-ai",
    version="0.1.0",
    description="Multi-modal deep learning pipeline for precision oncology: "
    "histopathology image analysis + genomic variant impact prediction.",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
