# Install openagent locally:
## git clone https://github.com/openagentsfoundation/openagents-sdk.git
## cd openagents-sdk
## pip install -e .

from setuptools import setup, find_packages
import pathlib

# Path to the project root (where requirements.txt lives)
HERE = pathlib.Path(__file__).parent

# Read requirements.txt and strip out blank lines or comments
req_path = HERE / "requirements.txt"
with req_path.open(encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f.read().splitlines()
        if line.strip() and not line.startswith("#")
    ]

# (Optional) Read long_description from README.md
long_description = ""
readme_path = HERE / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="openagents-sdk",
    version="0.1.0",
    description="OpenAgent SDK: tools and agents for building AI-driven assistants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="openagentsfoundation",
    author_email="community@openagentsfoundation.org",
    url="https://github.com/openagentsfoundation/openagents-sdk/openagent",
    packages=find_packages(exclude=["tests", "examples*"]),
    python_requires=">=3.12",
    install_requires=requirements,     # ← use requirements.txt here
    extras_require={
        # e.g. "dev": ["pytest>=6.0", "flake8"],
    },
    entry_points={
        # if you have console scripts, e.g.:
        # "console_scripts": [
        #     "openagent-cli = openagent.cli:main",
        # ],
    },
    license="MIT",
    include_package_data=True,
)
