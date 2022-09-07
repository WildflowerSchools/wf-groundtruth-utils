default_target := os() + "-" + arch()

build:
    poetry build

install target=default_target:
    #!/usr/bin/env bash

    if [[ "{{target}}" == "macos-aarch64" ]]; then
        poetry env use 3.10  # Or whatever version of Python you're using
        source "$( poetry env info --path )/bin/activate"
        pip install https://download.pytorch.org/whl/nightly/cpu/torch-1.13.0.dev20220906-cp310-none-macosx_11_0_arm64.whl
        pip install https://download.pytorch.org/whl/nightly/cpu/torchvision-0.14.0.dev20220906-cp310-cp310-macosx_11_0_arm64.whl
        pip install chumpy mmcv-full wf-pycocotools xtcocotools
        poetry install -vvv
    else
        export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
        poetry env use 3.10  # Or whatever version of Python you're using
        source "$( poetry env info --path )/bin/activate"
        pip install chumpy mmcv-full wf-pycocotools xtcocotools
        poetry install -vvv
    fi

fmt:
    autopep8 --aggressive --recursive --in-place ./groundtruth_utils/
