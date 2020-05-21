import io
import os

import boto3
from botocore.exceptions import ClientError
import json

from .interface import PlatformInterface
from .models.image import ImageList
from .models.job import Job, JobList
from ..aws.s3_util import download_fileobj_as_bytestream, split_s3_bucket_key


class Sagemaker(PlatformInterface):
    @staticmethod
    def fetch_job_by_name(job_name: str):
        try:
            sm_client = boto3.client('sagemaker')
            job_raw = sm_client.describe_labeling_job(
                LabelingJobName=job_name
            )
            if not job_raw:
                raise Exception("job not found")
            return job_raw
        except ClientError as e:
            print("Unexpected error: %s" % e)
            raise e

    def fetch_jobs(self, status='Completed', limit=0):
        status_title = status.title()
        valid_options = ['InProgress', 'Completed', 'Failed', 'Stopping', 'Stopped']

        if status_title not in valid_options:
            raise Exception("'status' must be one of %s" % valid_options)

        try:
            client = boto3.client('sagemaker')
            paginator = client.get_paginator('list_labeling_jobs')
            page_iterator = paginator.paginate(
                StatusEquals=status_title,
                SortBy='CreationTime'
            )

            result = []
            for page in page_iterator:
                for job_raw in page['LabelingJobSummaryList']:
                    result.append(Job.deserialize_sagemaker(job_raw))

                    if len(result) == limit:
                        break
                else:
                    continue
                break

            return JobList(jobs=result)
        except ClientError as e:
            print("Unexpected error: %s" % e)
            raise e

    def fetch_annotations(self, job_name: str, consolidate=True):
        job_raw = self.__class__.fetch_job_by_name(job_name)

        output_annotations_uri = job_raw['LabelingJobOutput']['OutputDatasetS3Uri']

        try:
            s3_client = boto3.client('s3')
            bytes_stream = download_fileobj_as_bytestream(s3_client, output_annotations_uri)
        except ClientError as e:
            print("Unexpected error fetching %s: %s" % (output_annotations_uri, e))
            raise e

        try:
            bytes_stream.seek(0)
            annotations_json = [json.loads(line) for line in bytes_stream.readlines()]
        finally:
            bytes_stream.close()

        if annotations_json is None:
            raise Exception("unable to download/process annotations data")

        return ImageList.deserialize_sagemaker(annotations_json)

    def generate_manifest(self, s3_images_uri: str, metadata: dict):
        folder_object_uris = self.__class__.list_images_in_s3_folder(s3_images_uri)

        output = ""
        bucket, _ = split_s3_bucket_key(s3_images_uri)

        try:
            fp = io.StringIO()
            for object_key in folder_object_uris:
                custom_metadata = {"externalId": os.path.basename(object_key)}
                if metadata:

                    custom_metadata.update(metadata)

                fp.write(json.dumps({
                    "source-ref": "https://{0}.s3.amazonaws.com/{1}".format(bucket, object_key),
                    "metadata": custom_metadata
                }) + "\n")
                output = fp.getvalue()
        finally:
            fp.close()

        return output

    def create_job(self, job_name='', attrs=None):
        pass
