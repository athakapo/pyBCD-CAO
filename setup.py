from setuptools import setup, find_packages

setup(
    name="pyBCD_CAO",
    version="1.0.0",
    description="Plug-n-play, Multi-robot Algorithm",
    author="Athanasios Kapoutsis",
    author_email="binary.point@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "numba>=0.55"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)