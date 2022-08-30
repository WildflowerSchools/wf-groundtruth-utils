import hashlib
import os
from pathlib import Path

from .log import logger
from .base import data_dir

import gdown


ALPHAPOSE_MXNET_WEIGHTS_ID = '1TTf8Ox-ECGXRAeX4cHYkEMBDVJEZgBL6'


# Thanks quantumSoup @ https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def sha256(fname):
    hash_md5 = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_checksum(id, dest_path):
    if id == ALPHAPOSE_MXNET_WEIGHTS_ID:
        return 'bf0e658795a89950b75268ac6bcaa0debf585e4e0265155abbd45529eecce626' == sha256(dest_path)

    return False


def download_weights(id=ALPHAPOSE_MXNET_WEIGHTS_ID):
    dest_path = None
    if id == ALPHAPOSE_MXNET_WEIGHTS_ID:
        dest_path = os.path.join(data_dir(), 'duc_se.params')

    if dest_path is None:
        return None

    if Path(dest_path).is_file() and validate_checksum(id, dest_path):
        return dest_path

    for ii in range(3):
        if ii > 0:
            logger.warn("Checksum failed, retrying download...")

        try:
            gdown.download(id=id, output=dest_path, quiet=False)
        except Exception as e:
            logger.error('Error at %s', 'file download', exc_info=e)
            return None

        if validate_checksum(id, dest_path):
            return dest_path

    logger.error('Unable to validate file checksum')
    return None
