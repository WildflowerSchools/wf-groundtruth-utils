build:
    poetry build

install:
    poetry install

fmt:
    autopep8 --aggressive --recursive --in-place ./groundtruth_utils/
