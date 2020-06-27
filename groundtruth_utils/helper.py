import os

from .platforms.sagemaker import Sagemaker
from .platforms.labelbox import Labelbox


def get_platform(platform: str):
    if platform == 'sagemaker':
        return Sagemaker()
    elif platform == 'labelbox':
        return Labelbox()
    else:
        raise Exception("'%s' invalid platform, must be sagemaker|labelbox")


def get_separated_file_name(image_name, idx):
    image_parts = os.path.splitext(os.path.basename(image_name))
    return "%s.%03d%s" % (image_parts[0], idx, image_parts[1])
