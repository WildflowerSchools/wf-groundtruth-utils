import json

from pydantic import BaseModel
from typing import List


class Classification(BaseModel):
    label: str
    value: List[str] = []
    raw_classification: dict = None

    @staticmethod
    def deserialize_labelbox(raw_classification):
        values = []
        if 'answers' in raw_classification:
            for answer in raw_classification['answers']:
                values.append(answer['title'])
        elif 'answer' in raw_classification:
            values.append(raw_classification['answer']['title'])

        return Classification(
            label=raw_classification['title'],
            value=values,
            raw_classification=raw_classification
        )


class ClassificationList(BaseModel):
    classifications: List[Classification] = []

    @staticmethod
    def deserialize_sagemaker(raw_classifications):
        pass

    @staticmethod
    def deserialize_labelbox(raw_labels):
        classifications = []
        for raw_label_metadata in raw_labels:
            raw_features = json.loads(raw_label_metadata['label'])
            if 'classifications' not in raw_features:
                continue

            for raw_classification in raw_features["classifications"]:
                raw_classification = Classification.deserialize_labelbox(raw_classification)
                classifications.append(raw_classification)

        # classifications = []
        # for idx, raw_classification in enumerate(raw_labels):
        #     classifications.append(Classification.deserialize_labelbox(raw_classification))

        return ClassificationList(
            classifications=classifications
        )
