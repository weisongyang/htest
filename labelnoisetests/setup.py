import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="noisetests",
    version="0.0.1",
    description="Small package for local regression.",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "locreg"},
    packages=setuptools.find_packages(where="noisetests"),
    python_requires=">=3.6",
)