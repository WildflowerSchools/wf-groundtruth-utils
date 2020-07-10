import abc

import boto3
from botocore.exceptions import ClientError

from ..aws.s3_util import list_object_keys_in_folder


class PlatformInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fetch_jobs') and
                callable(subclass.fetch_jobs) and
                hasattr(subclass, 'fetch_annotations') and
                callable(subclass.fetch_annotations) or
                hasattr(subclass, 'generate_image_set') and
                callable(subclass.generate_image_set) or
                NotImplemented)

    @staticmethod
    def list_images_in_s3_folder(s3_images_uri):
        try:
            s3_client = boto3.client('s3')
            return list_object_keys_in_folder(s3_client, s3_images_uri, image_filter=True)
        except ClientError as e:
            print("Unexpected error loading folder contents from '%s': %s" % (s3_images_uri, e))
            raise e

    @abc.abstractmethod
    def fetch_jobs(self, status: str, limit: int):
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_annotations(self, job_name: str, consolidate: bool,
                          filter_min_confidence: float, filter_min_labelers: int):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_manifest(self, s3_images_uri: str, metadata: dict):
        raise NotImplementedError

    @abc.abstractmethod
    def create_job(self, job_name='', attrs=None):
        raise NotImplemented
