import json
import math
import os

from labelbox import Client as LBClient, Project
import numpy as np

from .interface import PlatformInterface
from .labelbox_custom_pagination import LabelboxCustomPaginatedCollection
from .labelbox_queries import ALL_ANNOTATIONS_QUERY, ALL_PROJECTS_METRICS_QUERY
from .models.image import ImageList, Image
from .models.job import JobList, Job
from .utils.bounding_box import non_max_suppression_fast
from ..aws.s3_util import split_s3_bucket_key


class Labelbox(PlatformInterface):
    @staticmethod
    def fetch_raw_project_by_name(name: str):
        lb_client = LBClient()

        project_list = lb_client.get_projects(where=Project.name == name)
        myiter = iter(project_list)
        project = next(myiter)
        if not project:
            raise Exception("job not found")

        return project

    @staticmethod
    def fetch_raw_project_data_rows_by_name(name: str):
        project = Labelbox.fetch_raw_project_by_name(name)

        lb_client = LBClient()
        # TODO: loop with enumerator rather than building a list
        row_data = list(
            LabelboxCustomPaginatedCollection(
                lb_client, ALL_ANNOTATIONS_QUERY, {
                    "id": project.uid}, [
                    "project", "dataRows"]))

        return row_data

    @staticmethod
    def get_bboxes_by_label_from_row_data(row_data_instance: dict):
        bboxes_by_label = {}

        for tagger_label_collection in row_data_instance['labels']:
            label = json.loads(tagger_label_collection['label'])
            if not label:
                continue

            for feature in label['objects']:
                if 'bbox' in feature:
                    if feature['value'] not in bboxes_by_label:
                        bboxes_by_label[feature['value']] = []

                    bbox = feature['bbox']
                    x1 = bbox['left']
                    y1 = bbox['top']
                    x2 = x1 + bbox['width']
                    y2 = y1 + bbox['height']
                    bboxes_by_label[feature['value']].append((x1, y1, x2, y2))

        return bboxes_by_label

    def fetch_jobs(self, status: str, limit: int):
        lb_client = LBClient()
        projects = list(LabelboxCustomPaginatedCollection(lb_client, ALL_PROJECTS_METRICS_QUERY, {}, ["projects"]))

        def get_status_attrs(project):
            dataset_size = project["datasetSize"]
            consensus_coverage = project["autoAuditPercentage"]
            consensus_votes = project["autoAuditNumberOfLabels"]
            label_count = project["labelCount"]
            expected_labels = dataset_size + (consensus_votes - 1) * math.ceil(dataset_size * consensus_coverage)

            project_status = 'inprogress'
            if label_count == expected_labels:
                project_status = 'completed'

            status_attrs = {
                "expectedLabels": expected_labels,
                "status": project_status
            }
            project.update(status_attrs)
            return project

        detailed_projects = list(map(lambda p: get_status_attrs(p), projects))
        filtered_projects = list(
            filter(
                lambda project: 'status' in project and project['status'].lower() == status.lower(),
                detailed_projects))

        result = []
        for project in filtered_projects:
            result.append(Job.deserialize_labelbox(project))

        return JobList(jobs=result)

    def fetch_annotations(self, job_name: str):
        row_data = self.__class__.fetch_raw_project_data_rows_by_name(job_name)

        final_images = []
        for data in row_data:
            bboxes_by_label = self.__class__.get_bboxes_by_label_from_row_data(data)

            raw_annotations = []
            for k, v in bboxes_by_label.items():
                consolidated_boxes = non_max_suppression_fast(
                    np.asarray(v), max_annotations_per_object=len(
                        data["labels"])).tolist()
                for box in consolidated_boxes:
                    annotation = {
                        "label": k,
                        "left": box[0],
                        "top": box[1],
                        "width": box[2] - box[0],
                        "height": box[3] - box[1]}

                    raw_annotations.append(annotation)

            final_images.append(Image.deserialize_labelbox(data, raw_annotations))

        return ImageList(images=final_images)

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
