import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "PoissonReg",
    version = "0.0.1",
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
    package_dir = {".": "."},
    packages = setuptools.find_packages(where="."),
    python_requires = ">=3.6"
)
