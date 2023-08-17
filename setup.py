import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

__version__ = "0.0.1.13"

setuptools.setup(
    name = "poissonreg",
    version = __version__,
    author = "Moeen Nehzati",
    author_email = "moeen.nehzati@nyu.edu",
    description = "A package for fast sparse poisson regression",
    # long_description = long_description,
    # long_description_content_type = "text/markdown",
    url = "https://github.com/MoeenNehzati/PoissonReg",
    # project_urls = {
    #     "Bug Tracker": "package issues URL",
    # },
    classifiers = [
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = ">=3.6",
    install_requires=[
            'torch==2.0.1',
            'numpy==1.21.5',
            'tqdm==4.63.0',
            ],
      test_suite="poissonreg.tests",
)
