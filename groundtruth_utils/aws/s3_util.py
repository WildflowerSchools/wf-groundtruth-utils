import io


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
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    return find_bucket_key(s3_path)