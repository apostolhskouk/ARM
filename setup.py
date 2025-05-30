from setuptools import setup, find_packages

setup(
    name="arm_project",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # if your project has its own deps, list them here
    ],
    python_requires=">=3.7",
)
