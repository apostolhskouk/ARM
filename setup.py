from setuptools import setup, find_packages

setup(
    name="arm",
    version="0.1.0",
    # find the src package and all its subpackages
    packages=find_packages(include=["src", "src.*"]),
    # tell setuptools that those packages live under ./src
    package_dir={"src": "src"},
)