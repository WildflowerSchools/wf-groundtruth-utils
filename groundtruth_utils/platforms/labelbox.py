import json
import os

import boto3
from botocore.exceptions import ClientError

from .interface import PlatformInterface
from ..aws.s3_util import list_object_keys_in_folder, split_s3_bucket_key


class Labelbox(PlatformInterface):
    def fetch_jobs(self, status: str, limit: int):
        pass

    def fetch_annotations(self, job_name: str):
        pass

    def generate_manifest(self, s3_images_uri: str, metadata: dict):
        folder_object_uris = self.__class__.list_images_in_s3_folder(s3_images_uri)

        items = []
        bucket, _ = split_s3_bucket_key(s3_images_uri)
        for object_key in folder_object_uris:
            items.append({
                "externalId": os.path.basename(object_key),
                "imageUrl": "https://{0}.s3.amazonaws.com/{1}".format(bucket, object_key)
            })

        return json.dumps(items, indent=2)
