import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cd_network",
    version="0.1.4",
    author="Asaf Zorea",
    author_email="zoreasaf@gmail.com",
    description="A framework designed to calculate the output of neurons based on non-homogeneous Poisson processes "
    "and rate statistic calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nuniz/CoincidenceDetectionNetwork",
    project_urls={
        "Bug Tracker": "https://github.com/nuniz/CoincidenceDetectionNetwork/issues",
    },
    license="MIT",
    packages=setuptools.find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*", "tests.*"]
    ),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["cd_network = cd_network.run:main"]},
    install_requires=["numpy<2", "scipy", "matplotlib", "networkx"],
    extras_require={
        "dev": ["pytest", "check-manifest", "pre-commit"],
        "test": ["pytest", "coverage"],
    },
)
