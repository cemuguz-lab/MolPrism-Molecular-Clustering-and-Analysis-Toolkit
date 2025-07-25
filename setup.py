
from setuptools import setup, find_packages

setup(
    name="molprism",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'molprism=molprism.cli:main',
        ],
    },
    author="Cem UĞUZ",
    author_email="cemuguzlab@gmail.com",
    description="A Molecular Clustering and Analysis Toolkit",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cemuguz-lab/MolPrism",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires='>=3.8',
    install_requires=open('requirements.txt').read().splitlines(),
)
