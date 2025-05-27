from setuptools import setup

setup(
    name="geospatial_fm",
    version="0.1.0",
    description="MMSegmentation classes for geospatial-fm finetuning - Azure ML compatible",
    author="Paolo Fraccaro, Carlos Gomes, Johannes Jakubik",
    packages=["geospatial_fm"],
    license="Apache 2",
    install_requires=[
        "mmsegmentation==0.30.0",
        "urllib3==1.26.12",
        "rasterio",
        "tifffile",
        "einops",
        "timm==0.4.12",
        "tensorboard",
        "imagecodecs",
        # Added Azure ML specific dependencies
        "azure-ai-ml>=1.0.0",
        "azure-storage-blob>=12.0.0",
        "mlflow>=2.0.0",
        "torch>=1.12.0",
        "torchvision",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
