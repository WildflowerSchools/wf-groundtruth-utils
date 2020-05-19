from datetime import datetime
from pydantic import BaseModel
from typing import List


class Job(BaseModel):
    id: str
    name: str
    status: str
    labeled: int
    platform: str
    created_at: datetime
    updated_at: datetime
    raw: dict

    @staticmethod
    def exclude_raw():
        return {'raw'}

    @staticmethod
    def deserialize_sagemaker(raw):
        return Job(
            id=raw['LabelingJobArn'],
            name=raw['LabelingJobName'],
            status=raw['LabelingJobStatus'],
            labeled=raw['LabelCounters']['TotalLabeled'],
            platform='sagemaker',
            created_at=raw['CreationTime'],
            updated_at=raw['LastModifiedTime'],
            raw=raw
        )

    @staticmethod
    def deserialize_labelbox(raw):
        return Job(
            id=raw['id'],
            name=raw['name'],
            status=raw['status'],
            labeled=raw['labelCount'],
            platform='labelbox',
            created_at=raw['createdAt'],
            updated_at=raw['updatedAt'],
            raw=raw
        )


class JobList(BaseModel):
    jobs: List[Job]
