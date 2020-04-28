import boto3
from botocore.exceptions import ClientError
import json

from .interface import PlatformInterface
from .models.image import ImageList
from .models.job import Job, JobList
from ..aws.s3_util import download_fileobj_as_bytestream


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

    def fetch_annotations(self, job_name: str):
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
