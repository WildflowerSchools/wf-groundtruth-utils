from pydantic import BaseModel
from typing import List

class Classification(BaseModel):
    label: str
    value: str
    raw_classification: dict = None

    @staticmethod
    def deserialize_labelbox(raw_classification):
        return Classification(
            label=raw_classification['title'],
            value=raw_classification['answer']['title'],
            raw_classification=raw_classification
        )


class ClassificationList(BaseModel):
    classifications: List[Classification] = []

    @staticmethod
    def deserialize_sagemaker(raw_classifications):
        pass

    @staticmethod
    def deserialize_labelbox(raw_classifications):
        classifications = []
        for idx, raw_classification in enumerate(raw_classifications):
            classifications.append(Classification.deserialize_labelbox(raw_classification))

        return ClassificationList(
            classifications=classifications
        )
