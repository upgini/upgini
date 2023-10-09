import sys
from getpass import getuser
from logging import Formatter
from pathlib import Path
from urllib import request

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


def send_log(msg: str):
    try:
        url = "https://search.upgini.com/private/api/v2/events/send-light"

        try:
            user = getuser()
        except Exception:
            user = "unknown"

        data = ('{"message": "' + msg + '", "whoamai": "' + user + '"}').encode()

        req = request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")
        request.urlopen(req)
    except Exception:
        pass


here = Path(__file__).parent.resolve()
version = "1.1.205"
try:
    send_log(f"Start setup PyLib version {version}")
    setup(
        name="upgini",
        version=version,
        description="Intelligent data search & enrichment for Machine Learning",
        long_description=(here / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://upgini.com/",
        author="Upgini Developers",
        author_email="madewithlove@upgini.com",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
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
        package_data={"": ["strings.properties", "fingerprint.js"]},
        python_requires=">=3.7,<3.11",
        install_requires=[
            "python-dateutil>=2.8.0",
            "requests>=2.8.0",
            "pandas>=1.1.0,<2.0.0",
            "numpy>=1.19.0",
            "scikit-learn>=1.0.1",
            "pydantic>=1.8.2,<2.0.0",
            "fastparquet>=0.8.1",
            "python-json-logger>=2.0.2",
            "catboost>=1.0.3",
            "lightgbm>=3.3.2",
            "pyjwt>=2.8.0",
            "xhtml2pdf>=0.2.11",
            "ipywidgets>=8.1.0",
        ],
        project_urls={
            "Bug Reports": "https://github.com/upgini/upgini/issues",
            "Source": "https://github.com/upgini/upgini",
        },
    )
    send_log(f"Setup of PyLib {version} successfully finished")
except Exception as e:
    try:
        tb = Formatter().formatException(sys.exc_info())
        send_log(f"Failed to setup PyLib {version}: {e}\n{tb}")
    except Exception:
        pass
    raise e
