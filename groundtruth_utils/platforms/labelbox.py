import copy
import json
import math
import os
import uuid

import cv2 as cv
from labelbox import Client as LBClient, exceptions as LBExceptions
import numpy as np
from pycocotools.coco import COCO

from .interface import PlatformInterface
from .labelbox_api import LabelboxAPI
from .labelbox_coco import coco_annotation_to_labelbox
from .labelbox_custom_pagination import LabelboxCustomPaginatedCollection
from .labelbox_queries import ALL_PROJECTS_METRICS_QUERY
from .models.annotation import AnnotationTypes
from .models.image import ImageList, Image
from .models.job import JobList, Job
from .utils.bounding_box import non_max_suppression_fast
from .utils.util import random_id
from ..coco.models.annotation import KeypointAnnotation as CocoKeypointAnnotation
from ..log import logger
from ..aws.s3_util import split_s3_bucket_key

MAL_NAMESPACE = uuid.UUID('ac64f513-98cd-4c82-8a5b-2c9842240659')


class Labelbox(PlatformInterface):
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
                annotation.id = None
                consolidated_annotation.left = box[0]
                consolidated_annotation.top = box[1]
                consolidated_annotation.width = box[2] - box[0]
                consolidated_annotation.height = box[3] - box[1]
                consolidated_annotations.append(consolidated_annotation)

        # Consolidate points
        for k, v in keypoint_annotations_by_label.items():
            annotation = v['annotation']
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
                annotation.id = None
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
        row_data = LabelboxAPI.fetch_raw_project_data_rows_by_name(job_name)

        final_images = []
        for raw_data_row in row_data:
            image = Image.deserialize_labelbox(raw_data_row)

            if consolidate:
                image.annotations = self.__class__.consolidate_annotations(image.annotations)

            final_images.append(image)

        return ImageList(images=final_images)

    def fetch_images(self, job_name: str):
        row_data = LabelboxAPI.fetch_all_project_images(job_name)

        final_images = []
        for raw_data_row in row_data:
            image = Image.deserialize_labelbox(raw_data_row)
            final_images.append(image)

        return final_images

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

    def create_job(self, job_name='', attrs=None):
        if attrs is None:
            attrs = {}

        if 'ontology_json' not in attrs or 'dataset_id' not in attrs:
            raise Exception("create_job requires an 'ontology_json' and 'dataset_id' attributes")

        labeling_frontend_id = LabelboxAPI.get_image_labeling_frontened_id()

        # Attempt to fetch dataset, will throw an exception if not founds
        LabelboxAPI.fetch_raw_dataset_by_id(attrs['dataset_id'])

        lb_client = LBClient()
        organization = lb_client.get_organization()
        project = lb_client.create_project(name=job_name)

        if project is None:
            raise Exception("Could not create project '%s'" % job_name)

        try:
            LabelboxAPI.attach_dataset(project.uid, attrs['dataset_id'], labeling_frontend_id)
            LabelboxAPI.configure_interface_for_project(
                project.uid, labeling_frontend_id, organization.uid, attrs['ontology_json'])
        except Exception as e:
            logger.error('Error at %s', 'project setup', exc_info=e)
            LabelboxAPI.delete_project(project.uid)

        return project

    def upload_coco_dataset(self, job_name, coco_annotation_file):
        project = LabelboxAPI.fetch_raw_project_by_name(job_name)
        ontology = LabelboxAPI.get_project_ontology(project.uid)

        coco = COCO(coco_annotation_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        for img_id in img_ids:
            img = coco.loadImgs([img_id])

            data_row = None
            for dataset in project.datasets():
                try:
                    data_row = dataset.data_row_for_external_id(img[0]['file_name'])
                except LBExceptions.ResourceNotFoundError:
                    continue

            if data_row is None:
                logger.warn("Unable to find %s in project dataset" % (img[0]['file_name']))
                continue

            feature_ids = []
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=cat_ids))
            logger.info("Adding features to image/external_id %s" % (img[0]['file_name']))
            for annotation in annotations:
                keypoint_annotation = CocoKeypointAnnotation(**annotation)

                labels = coco_annotation_to_labelbox(keypoint_annotation, ontology)
                if len(labels) > 0:
                    for label in labels:
                        # Lots of steps below just to create a label in Labelbox, phew
                        new_object_feature = LabelboxAPI.create_new_object_feature(
                            schema_id=label['schema_id'],
                            project_id=project.uid,
                            datarow_id=data_row.uid,
                            content={'geometry': label['geo_json']})
                        feature_ids.append(new_object_feature['id'])

                        if label['nested_classification_feature'] is not None:
                            nested_feature = LabelboxAPI.create_new_nested_classification_feature(
                                parent_feature_id=new_object_feature['id'],
                                question_schema_id=label['nested_classification_feature']['question_schema_id'],
                                options_schema_ids=label['nested_classification_feature']['options_schema_ids']
                            )
                            feature_ids.append(nested_feature['result']['id'])
                            if len(nested_feature['descendants']) > 0:
                                for descendent in nested_feature['descendants']:
                                    feature_ids.append(descendent['id'])

                            classification_feature = LabelboxAPI.update_classification_options(
                                question_feature_id=nested_feature.id,
                                option_schema_ids=label['nested_classification_feature']['options_schema_ids']
                            )
                            if len(classification_feature['descendants']) > 0:
                                for descendent in classification_feature['descendants']:
                                    feature_ids.append(descendent['id'])

            logger.info("Generating labels from features for image/external_id %s" % (img[0]['file_name']))
            LabelboxAPI.create_label_from_features(
                project_id=project.uid,
                datarow_id=data_row.uid,
                feature_ids=feature_ids
            )

    def delete_unlabeled_features(self, job_name):
        project = LabelboxAPI.fetch_raw_project_by_name(job_name)
        data_rows = LabelboxAPI.fetch_all_project_images(job_name)

        for data_row in data_rows:
            features = LabelboxAPI.fetch_all_features_for_datarow(project.uid, data_row['id'])

            logger.info('Deleting features for dataRow %s' % data_row['id'])
            for feature in features:
                if feature['label'] is None:
                    logger.info('Deleting feature %s...' % feature['id'])
                    LabelboxAPI.delete_feature(feature['id'])
                else:
                    logger.info(
                        'NOT deleting feature %s. Feature has associated label: %s' %
                        (feature['id'], feature['label']['id']))

    def generate_mal_ndjson(self, job_name, annotations, deleteFeatures=False):
        """Annotations object format: [{'image': Coco.Image, 'annotations': Coco.Annotation}]"""

        if deleteFeatures:
            self.delete_unlabeled_features(job_name)

        project = LabelboxAPI.fetch_raw_project_by_name(job_name)
        ontology = LabelboxAPI.get_project_ontology(project.uid)

        mal_records = []
        for annotation in annotations:
            data_row = None
            for dataset in project.datasets():
                try:
                    data_row = dataset.data_row_for_external_id(annotation['image'].file_name)  # annotation['image'].file_name)
                except LBExceptions.ResourceNotFoundError:
                    continue

            if data_row is None:
                logger.warn("Unable to find %s in project dataset" % (annotation['image'].file_name))
                continue

            for coco_annotation in annotation['annotations']:
                labels = coco_annotation_to_labelbox(coco_annotation, ontology)
                for label in labels:
                    mal_records.append({
                        **{
                            "uuid": str(uuid.uuid5(MAL_NAMESPACE, "%s_%s_%s" % (label['schema_id'], data_row.uid, json.dumps(label['labelbox_geom'])))),
                            "schemaId": label['schema_id'],
                            "dataRow": {
                                "id": data_row.uid,
                            },
                        },
                        **label['labelbox_geom']
                    })

        return mal_records

    def create_mal_import_job(self, job_name, mal_file_url):
        project = LabelboxAPI.fetch_raw_project_by_name(job_name)

        import_id = random_id(32)
        logger.info("Creating import job: %s" % import_id)
        LabelboxAPI.create_mal_import_request(project.uid, import_id, mal_file_url)

        return import_id

    def get_status_mal_import_job(self, job_name, import_id):
        project = LabelboxAPI.fetch_raw_project_by_name(job_name)

        return LabelboxAPI.get_status_mal_import_request(project.uid, import_id)
