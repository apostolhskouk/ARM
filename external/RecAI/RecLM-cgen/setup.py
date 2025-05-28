from setuptools import setup, find_packages

# Read the requirements.txt into install_requires
with open("requirements.txt", "r") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="rec_lm_cgen",
    version="0.1.0",
    description="RecLM-cgen: constrained generative recommendation with LLMs",
    author="Shenzhen University / Microsoft",
    packages=find_packages(),          # automatically find train_utils, unirec, etc.
    install_requires=requirements,     # from requirements.txt
    python_requires=">=3.7",
    entry_points={
        "console_scripts": {
            # optional: expose any CLI entrypoints, e.g.
            "rec-lm-cgen-serve=cli_serve:main",
        },
    },
)
