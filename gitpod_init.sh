#!/bin/sh
pyenv local 3.8.12
pyenv shell 3.8.12
pip3 install -r requirements.txt
pip3 install catboost
git pull
python3 setup.py install