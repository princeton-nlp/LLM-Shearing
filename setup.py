import io
from setuptools import setup, find_packages 
import pathlib
import pkg_resources

with pathlib.Path('requirement.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name='llmshearing',
    packages=["llmshearing"],
    version='0.1',
    description='LLM Shearing',
    author='Mengzhou Xia',
    url='https://github.com/princeton-nlp/LLM-Shearing',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)

# auto-gptq cannot be installed