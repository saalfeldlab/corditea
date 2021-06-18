import os
from setuptools import find_packages, setup

NAME = "corditea"
DESCRIPTION = "Utility nodes for gunpowder"
URL = "https://github.com/saalfeldlab/corditea"
EMAIL = "heinrichl@janelia.hhmi.org"
AUTHOR = "Larissa Heinrich"
REQUIRES_PYTHON = ">=3.6"
VERSION = "0.1.dev1"

REQUIRED = [
    "numpy",
    "gunpowder",
    "scipy"
]

EXTRAS = {
}

DEPENDENCY_LINKS = [
]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = "\n" + f.read()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    entry_points={},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
    include_package_data=True,
    license="BSD-2-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
