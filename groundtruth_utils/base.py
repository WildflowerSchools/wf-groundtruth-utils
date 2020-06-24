import os
import platform


# data_dir_default & data_dir inspired by mxnet's method of downloading and caching models: https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/base.py
def data_dir_default():
    """
    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'wf_groundtruth_utils')
    else:
        return os.path.join(os.path.expanduser("~"), '.wf_groundtruth_utils')


def data_dir():
    """
    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('WF_GROUNDTRUTH_HOME', data_dir_default())
