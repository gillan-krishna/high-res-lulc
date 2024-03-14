import os
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
# here = os.path.abspath(os.path.dirname("__file__"))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except Exception:
    REQUIRED = []

setup(
    name="open_earth_map",
    version="0.1",
    description="Code for the Highres LULC project",
    author="Gillan Krishna",
    author_email="gillan@satsure.co",
    packages=find_packages(exclude=("data", "pics")),
    install_requires=REQUIRED,
)