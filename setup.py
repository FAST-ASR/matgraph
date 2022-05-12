import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matgraph",
    version="0.1.0",
    author="Lucas Ondel",
    author_email="lucas.ondel@gmail.com",
    description="Matrix-based graph manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FAST-ASR/matgraph",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./"),
    python_requires=">=3.7",
)


