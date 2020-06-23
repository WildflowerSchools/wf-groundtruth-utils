from datetime import datetime
import json

from labelbox import Client as LBClient, Dataset, Project
from .labelbox_custom_pagination import LabelboxCustomPaginatedCollection
from .labelbox_queries import ALL_ANNOTATIONS_QUERY, ATTACH_DATASET_AND_FRONTEND, ALL_FEATURES_FOR_DATAROW_QUERY, ALL_PROJECT_IMAGES_QUERY, CREATE_LABEL_FROM_FEATURES, CREATE_MAL_IMPORT_REQUEST, CREATE_NEW_NESTED_CLASSIFICATION_FEATURE, CREATE_NEW_OBJECT_FEATURE, CONFIGURE_INTERFACE_FOR_PROJECT, DELETE_FEATURE, DELETE_PROJECT, GET_IMAGE_LABELING_FRONTEND_ID, GET_PROJECT_ONTOLOGY, GET_STATUS_MAL_IMPORT_REQUEST, UPDATE_CLASSIFICATION_OPTIONS
from ..log import logger


class LabelboxAPI(object):
    @staticmethod
    def delete_project(project_id: str):
        lb_client = LBClient()
        lb_client.execute(DELETE_PROJECT, {'projectId': project_id})

    @staticmethod
    def fetch_raw_dataset_by_id(uid: str):
        lb_client = LBClient()

        dataset_list = lb_client.get_datasets(where=Dataset.uid == uid)
        myiter = iter(dataset_list)
        dataset = next(myiter)
        if not dataset:
            raise Exception("dataset not found")

        return dataset

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
        project = LabelboxAPI.fetch_raw_project_by_name(name)

        lb_client = LBClient()
        # TODO: loop with enumerator rather than building a list
        row_data = list(
            LabelboxCustomPaginatedCollection(
                lb_client, ALL_ANNOTATIONS_QUERY, {
                    "projectId": project.uid}, [
                    "project", "dataRows"]))

        return row_data

    @staticmethod
    def fetch_all_project_images(name: str):
        project = LabelboxAPI.fetch_raw_project_by_name(name)

        lb_client = LBClient()
        row_data = list(
            LabelboxCustomPaginatedCollection(
                lb_client, ALL_PROJECT_IMAGES_QUERY, {
                    "projectId": project.uid}, [
                    "project", "dataRows"]))

        return row_data

    @staticmethod
    def get_image_labeling_frontened_id():
        lb_client = LBClient()
        results = lb_client.execute(GET_IMAGE_LABELING_FRONTEND_ID)
        return results['labelingFrontends'][0]['id']

    @staticmethod
    def get_project_ontology(project_id: str):
        lb_client = LBClient()
        project = lb_client.execute(GET_PROJECT_ONTOLOGY, {'projectId': project_id})['project']
        return project['ontology']['normalized']

    @staticmethod
    def create_new_object_feature(schema_id: str, project_id: str, datarow_id: str,
                                  content: dict, label=None, seconds_spent=0):
        lb_client = LBClient()

        params = {
            "schemaId": schema_id,
            "projectId": project_id,
            "label": label,
            "dataRowId": datarow_id,
            "secondsSpent": seconds_spent,
            "content": content
        }

        logger.info("Executing CREATE_NEW_OBJECT_FEATURE with: %s" % (params))
        feature = lb_client.execute(CREATE_NEW_OBJECT_FEATURE, params)

        return feature['createObjectFeature']

    @staticmethod
    def create_new_nested_classification_feature(
            parent_feature_id: str, question_schema_id: str, options_schema_ids=[], seconds_spent=0):
        lb_client = LBClient()

        params = {
            "parentFeatureId": parent_feature_id,
            "questionSchemaId": question_schema_id,
            "optionSchemaIds": options_schema_ids,
            "secondsSpent": seconds_spent
        }
        logger.info("Executing CREATE_NEW_NESTED_CLASSIFICATION_FEATURE with: %s" % (params))
        feature = lb_client.execute(CREATE_NEW_NESTED_CLASSIFICATION_FEATURE, params)

        return feature['objectFeature']['createClassificationFeature']['setOptions']

    @staticmethod
    def update_classification_options(question_feature_id: str, option_schema_ids=[], additional_seconds_spent=0):
        lb_client = LBClient()

        params = {
            "questionFeatureId": question_feature_id,
            "optionSchemaIds": option_schema_ids,
            "additionalSecondsSpent": additional_seconds_spent
        }

        logger.info("Executing UPDATE_CLASSIFICATION_OPTIONS with: %s" % (params))
        feature = lb_client.execute(UPDATE_CLASSIFICATION_OPTIONS, params)

        return feature["classificationFeature"]["setOptions"]

    @staticmethod
    def create_label_from_features(project_id: str, datarow_id: str, feature_ids=[]):
        lb_client = LBClient()

        params = {
            "secondsSpent": 0,
            "featureIds": feature_ids,
            "dataRowId": datarow_id,
            "projectId": project_id
        }
        logger.info("Executing CREATE_LABEL_FROM_FEATURES with: %s" % (params))
        output = lb_client.execute(CREATE_LABEL_FROM_FEATURES, params)

        return output['createLabelFromFeatures']

    @staticmethod
    def fetch_all_features_for_datarow(project_id: str, datarow_id: str):
        lb_client = LBClient()

        features = list(
            LabelboxCustomPaginatedCollection(
                lb_client, ALL_FEATURES_FOR_DATAROW_QUERY, {
                    "projectId": project_id,
                    "dataRowId": datarow_id
                }, ["project", "featuresForDataRow"]))

        return features

    @staticmethod
    def delete_feature(feature_id: str):
        lb_client = LBClient()
        result = lb_client.execute(DELETE_FEATURE, {
            'featureId': feature_id
        })

        return result['deleteFeature']

    @staticmethod
    def attach_dataset(project_uid: str, dataset_id: str, labeling_frontend_id: str):
        lb_client = LBClient()
        lb_client.execute(ATTACH_DATASET_AND_FRONTEND, {
            'projectId': project_uid,
            'datasetId': dataset_id,
            'labelingFrontendId': labeling_frontend_id,
            'date': datetime.now()
        })

    @staticmethod
    def configure_interface_for_project(project_uid: str, labeling_frontend_id: str,
                                        organization_uid: str, customization_options=[]):
        lb_client = LBClient()
        lb_client.execute(CONFIGURE_INTERFACE_FOR_PROJECT, {
            'projectId': project_uid,
            'customizationOptions': json.dumps(customization_options),
            'labelingFrontendId': labeling_frontend_id,
            'organizationId': organization_uid
        })

    @staticmethod
    def create_mal_import_request(project_uid: str, import_id: str, file_url: str):
        lb_client = LBClient()
        output = lb_client.execute(CREATE_MAL_IMPORT_REQUEST, {
            'projectId': project_uid,
            'importName': import_id,
            'fileUrl': file_url
        })

        return output['createBulkImportRequest']['id']

    @staticmethod
    def get_status_mal_import_request(project_uid: str, import_id: str):
        lb_client = LBClient()
        output = lb_client.execute(GET_STATUS_MAL_IMPORT_REQUEST, {
            'projectId': project_uid,
            'importName': import_id
        })

        return output
