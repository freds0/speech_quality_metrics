from setuptools import setup, find_packages

setup(
    name="speech_quality_metrics",
    version="0.0.1",
    author="Frederico S. Oliveira",
    description="Speech Quality Metrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/freds0/speech_quality_metrics",
    packages=find_packages("src"),  # Correção aqui: especifica o diretório src
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchaudio",
        "resemblyzer",
        "jiwer",
        "pydub",
        "chardet",
        "speechbrain",
        "tqdm",
        "transformers",
        "pandas",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Homepage": "https://github.com/freds0/speech_quality_metrics",
        "Issues": "https://github.com/freds0/speech_quality_metrics/issues",
    },
)
