build:
    python setup.py install

install-dev:
    pip install -e .[development]

fmt:
    autopep8 --aggressive --recursive --in-place ./groundtruth_utils/
