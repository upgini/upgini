from pathlib import Path

from setuptools import find_packages, setup

# To build:
# python setup.py sdist
# python setup.py bdist_wheel
#
# To install:
# python setup.py install
#
# To register (only once):
# python setup.py register
#
# To upload:
# python setup.py sdist upload
# python setup.py bdist_wheel upload

here = Path(__file__).parent.resolve()
setup(
    name="upgini",
    version="0.10.0a70",
    description="Features search library for supervised machine learning on tabular data",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://upgini.com/",
    author="Upgini Developers",
    author_email="madewithlove@upgini.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    license="BSD 3-Clause License",
    keywords=["data science", "machine learning", "data mining", "automl", "data search"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7,<4",
    install_requires=[
        "python-dateutil>=2.8.0",
        "requests>=2.8.0",
        "pandas>=1.1.0",
        "numpy>=1.19.0",
        "imbalanced-learn>=0.9.0",
        "pydantic>=1.8.2",
        "pyarrow>=5.0.0",
        "fastparquet>=0.7.1",
        "yaspin>=2.1.0",
        "python-json-logger>=2.0.2",
    ],
    project_urls={
        "Bug Reports": "https://github.com/upgini/upgini/issues",
        "Source": "https://github.com/upgini/upgini",
    },
)
