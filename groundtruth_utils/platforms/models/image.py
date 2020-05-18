from pydantic import BaseModel
from typing import List

from .annotation import Annotation, AnnotationList
from .classification import Classification, ClassificationList


class Image(BaseModel):
    id: str
    external_id: str
    url: str
    width: int = None
    height: int = None
    annotations: List[Annotation]
    classifications: List[Classification] = []

    @staticmethod
    def deserialize_sagemaker(raw):
        raw_output = None
        raw_metadata = None
        for key in raw.keys():
            if key == "source-ref":
                continue
            elif "metadata" in key:
                raw_metadata = raw[key]
            else:
                raw_output = raw[key]

        return Image(
            url=raw['source-ref'],
            width=raw_output['image_size'][0]['width'],
            height=raw_output['image_size'][0]['height'],
            annotations=AnnotationList.deserialize_sagemaker(raw_output['annotations'], raw_metadata).annotations
        )

    @staticmethod
    def deserialize_labelbox(raw_data_row):
        return Image(
            id=raw_data_row['id'],
            external_id=raw_data_row['externalId'],
            url=raw_data_row['rowData'],
            annotations=AnnotationList.deserialize_labelbox(raw_data_row['labels']).annotations,
            classifications=ClassificationList.deserialize_labelbox(raw_data_row['labels']).classifications
        )


class ImageList(BaseModel):
    images: List[Image]

    @staticmethod
    def deserialize_sagemaker(raw_list):
        deserialized_images = []

        for raw_record in raw_list:
            deserialized_images.append(Image.deserialize_sagemaker(raw_record))

        return ImageList(
            images=deserialized_images
        )
