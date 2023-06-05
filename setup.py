from setuptools import setup, find_packages

setup(
    name="dowg",
    version="0.12",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    author="Ahmed Khaled",
    author_email="akregeb@gmail.com",
    description="DoWG optimizer for PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rka97/dowg",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
