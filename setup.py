from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """Get the list of requirements from a requirements file.

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of requirements.
    """
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-e')]

setup(
    name="firstmlproject",
    version="0.1",
    author="Niral Patel",
    author_email="nir64.au@gmail.com",
    description="A simple first ML project",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)