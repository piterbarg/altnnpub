import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="altnnpub",  # Replace with your own username
    version="0.0.1",
    author="Vladimir V. Piterbarg",
    author_email="personal@megabarg.com",
    description="Implementation of gSS anf fTT algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/piterbarg/altnnpub/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)