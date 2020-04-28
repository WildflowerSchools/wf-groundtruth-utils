from .platforms.sagemaker import Sagemaker
from .platforms.labelbox import Labelbox


def get_platform(platform: str):
    if platform == 'sagemaker':
        return Sagemaker()
    elif platform == 'labelbox':
        return Labelbox()
    else:
        raise Exception("'%s' invalid platform, must be sagemaker|labelbox")
