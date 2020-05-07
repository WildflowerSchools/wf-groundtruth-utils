from pydantic import BaseModel
from typing import List

from .annotation import Annotation, AnnotationList


class Image(BaseModel):
    id: str
    external_id: str
    url: str
    width: int = None
    height: int = None
    annotations: List[Annotation]

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
    def deserialize_labelbox(data, annotations):
        return Image(
            id=data['id'],
            external_id=data['externalId'],
            url=data['rowData'],
            annotations=AnnotationList.deserialize_labelbox(annotations).annotations
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
