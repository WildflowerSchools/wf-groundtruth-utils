import copy
import json
import math
import os

import cv2 as cv
from labelbox import Client as LBClient, Project
import numpy as np

from .interface import PlatformInterface
from .labelbox_custom_pagination import LabelboxCustomPaginatedCollection
from .labelbox_queries import ALL_ANNOTATIONS_QUERY, ALL_PROJECTS_METRICS_QUERY
from .models.annotation import AnnotationTypes
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

    # @staticmethod
    # def get_annotations_by_label_from_row_data(type: str, row_data_instance: dict):
    #     annotations_by_label = {}
    #
    #     for tagger_label_collection in row_data_instance['labels']:
    #         label = json.loads(tagger_label_collection['label'])
    #         if not label:
    #             continue
    #
    #         for feature in label['objects']:
    #             classifications = feature['classifications'] if 'classifications' in feature else []
    #             if type == 'bbox' and 'bbox' in feature:
    #                 if feature['value'] not in annotations_by_label:
    #                     annotations_by_label[feature['value']] = {
    #                         "classifications": [],
    #                         "boxes": []
    #                     }
    #
    #                 bbox = feature['bbox']
    #                 x1 = bbox['left']
    #                 y1 = bbox['top']
    #                 x2 = x1 + bbox['width']
    #                 y2 = y1 + bbox['height']
    #                 annotations_by_label[feature['value']]["boxes"].append((x1, y1, x2, y2))
    #                 annotations_by_label[feature['value']]["classifications"].extend(classifications)
    #
    #             elif type == 'keypoint' and 'point' in feature:
    #                 if feature['value'] not in annotations_by_label:
    #                     annotations_by_label[feature['value']] = {
    #                         "classifications": [],
    #                         "keypoints": []
    #                     }
    #
    #                 point = feature['point']
    #                 x = point['x']
    #                 y = point['y']
    #                 annotations_by_label[feature['value']]["keypoints"].append((x, y))
    #                 annotations_by_label[feature['value']]["classifications"].extend(classifications)
    #
    #     return annotations_by_label

    @staticmethod
    def consolidate_annotations(annotations: list):
        labelers = []
        box_annotations_by_label = {}
        keypoint_annotations_by_label = {}

        # First find and group all annotations that should be consolidated
        for annotation in annotations:
            labeler = annotation.raw_metadata['createdBy']['email']
            if labeler not in labelers:
                labelers.append(labeler)
            # Build list of bounding box sets
            if annotation.type == AnnotationTypes.TYPE_BOUNDING_BOX:
                if annotation.label not in box_annotations_by_label:
                    box_annotations_by_label[annotation.label] = {'annotation': annotation, 'bboxes': []}

                x1 = annotation.left
                y1 = annotation.top
                x2 = x1 + annotation.width
                y2 = y1 + annotation.height
                box_annotations_by_label[annotation.label]['bboxes'].append((x1, y1, x2, y2))

            # Build list of keypoint sets
            elif annotation.type == AnnotationTypes.TYPE_KEYPOINT:
                if annotation.label not in keypoint_annotations_by_label:
                    keypoint_annotations_by_label[annotation.label] = {'annotation': annotation, 'points': []}

                x = annotation.x
                y = annotation.y

                keypoint_annotations_by_label[annotation.label]['points'].append((x, y))

        consolidated_annotations = []

        # Consolidate boxes
        for k, v in box_annotations_by_label.items():
            annotation = v['annotation']
            consolidated_boxes = non_max_suppression_fast(
                np.asarray(v["bboxes"]), max_annotations_per_object=len(
                    labelers)).tolist()

            for box in consolidated_boxes:
                consolidated_annotation = copy.deepcopy(annotation)
                consolidated_annotation.left = box[0]
                consolidated_annotation.top = box[1]
                consolidated_annotation.width = box[2] - box[0]
                consolidated_annotation.height = box[3] - box[1]
                consolidated_annotations.append(consolidated_annotation)

        # Consolidate points
        for k, v in keypoint_annotations_by_label.items():
            annotation = v['annotation']
            # TODO: Improve point consolidation, likely want to do some sort of conslida
            #consolidated_keypoint =  np.mean(np.asarray(v["points"]))
            points = []
            if len(v["points"]) == 1:
                points = v["points"]
            elif len(v["points"]) > 1:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                flags = cv.KMEANS_RANDOM_CENTERS
                num_clusters = round(len(v["points"]) / len(labelers))
                _, _, cluster_points = cv.kmeans(np.float32(v["points"]), num_clusters, None, criteria, 10, flags)
                points = cluster_points.tolist()

            for point in points:
                consolidated_annotation = copy.deepcopy(annotation)
                consolidated_annotation.x = point[0]
                consolidated_annotation.y = point[1]
                consolidated_annotations.append(consolidated_annotation)

        return consolidated_annotations

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

    def fetch_annotations(self, job_name: str, consolidate=True):
        row_data = self.__class__.fetch_raw_project_data_rows_by_name(job_name)

        final_images = []
        for raw_data_row in row_data:
            image = Image.deserialize_labelbox(raw_data_row)

            if consolidate:
                image.annotations = self.__class__.consolidate_annotations(image.annotations)

            final_images.append(image)

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
