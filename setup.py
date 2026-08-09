from setuptools import setup, find_packages

setup(
    name="thinkncollab-scribe",
    version="1.0.0",
    description="Official ThinkNCollab Hinglish (Hindi + Indian English) Speech-to-Text ASR Script & Package",
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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
