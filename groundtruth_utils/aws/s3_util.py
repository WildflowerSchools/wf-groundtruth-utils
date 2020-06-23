import io
import os
import re

import boto3
from botocore.exceptions import ClientError

from ..log import logger


def download_fileobj_as_bytestream(s3_client, object_uri):
    bytes_stream = io.BytesIO()
    bucket_name, key_name = split_s3_bucket_key(object_uri)
    s3_client.download_fileobj(bucket_name, key_name, bytes_stream)
    return bytes_stream


#  Thanks https://stackoverflow.com/questions/4993439/how-can-i-access-s3-files-in-python-using-urls
def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])
    if bucket.endswith('.s3.amazonaws.com'):
        bucket = bucket.split('.')[0]
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    elif s3_path.startswith('https://'):
        s3_path = s3_path[8:]
    return find_bucket_key(s3_path)


def list_object_keys_in_folder(s3_client, folder_uri, filter_regex=None, image_filter=False):
    bucket_name, key_name = split_s3_bucket_key(folder_uri)

    active_filter_regex = None
    if image_filter:
        active_filter_regex = r".*\.(gif|jpe?g|tiff|png|webp|bmp)$"
    elif filter_regex:
        active_filter_regex = r"%s" % filter_regex

    paginator = s3_client.get_paginator('list_objects')
    iterator = paginator.paginate(Bucket=bucket_name, Prefix=key_name)
    bucket_object_list = []
    for page in iterator:
        if "Contents" in page:
            for key in page["Contents"]:
                key_string = key["Key"]

                match = True
                if active_filter_regex is not None:
                    match = bool(re.match(active_filter_regex, key_string, re.IGNORECASE))
                if match:
                    bucket_object_list.append(key_string)

    return bucket_object_list


def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logger.error(e)
        return None

    # The response contains the presigned URL
    return response


def upload_file_to_bucket(file_path, bucket, object_name=None, meta_data={}):
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket, object_name, ExtraArgs={'Metadata': meta_data})
    except ClientError as e:
        logger.error(e)
        return None

    return create_presigned_url(bucket, object_name)
