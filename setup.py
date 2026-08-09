from setuptools import setup, find_packages

setup(
    name="thinkncollab-scribe",
    version="1.0.1",
    description="Official ThinkNCollab Hinglish (Hindi + Indian English) Speech-to-Text ASR Model & CLI Package",
    author="ThinkNCollab AI Team",
    url="https://github.com/Ramansingh9948/ThinkNCollab_TRANSCRIPT",
    py_modules=["thinkncollab_whisper"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "sentencepiece",
        "torch",
        "ctranslate2"
    ],
    entry_points={
        "console_scripts": [
            "thinkncollab-scribe = thinkncollab_whisper:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="speech-to-text asr whisper hinglish hindi indian-english speech-recognition noise-reduction",
    python_requires=">=3.8",
)
