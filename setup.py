from setuptools import setup, find_packages

setup(
    name="dualcore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "seaborn>=0.12.0",
    ],
    author="Antigravity",
    description="A cognitive architecture module based on dual axes.",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
