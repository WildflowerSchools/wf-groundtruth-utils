import json

from pydantic import BaseModel
from typing import List


class AnnotationTypes:
    TYPE_BOUNDING_BOX = "BoundingBox"
    TYPE_KEYPOINT = "Keypoint"
    TYPE_UNKNOWN = "Unknown"


class Annotation(BaseModel):
    confidence: float = None
    type: str
    raw_annotation: dict
    raw_metadata: dict = None
    raw_metadata_annotation_idx: int = None

    @staticmethod
    def deserialize_sagemaker(raw_annotation, raw_metadata, idx):
        bounding_box_fingerprint = ['class_id', 'width', 'height', 'left', 'top']
        if all(attr in raw_annotation for attr in bounding_box_fingerprint):
            return BoundingBoxAnnotation.deserialize_sagemaker(raw_annotation, raw_metadata, idx)
        else:
            return Annotation(
                confidence=raw_metadata["confidence"],
                raw_annotation=raw_annotation,
                raw_metadata=raw_metadata,
                raw_metadata_annotation_idx=idx,
                type=AnnotationTypes.TYPE_UNKNOWN
            )

    @staticmethod
    def deserialize_labelbox(raw_label_metadata, raw_feature):
        bounding_box_fingerprint = ['title', 'width', 'height', 'left', 'top']
        keypoint_fingerprint = ['title', 'point']

        if all(attr in raw_feature for attr in bounding_box_fingerprint):
            return BoundingBoxAnnotation.deserialize_labelbox(raw_label_metadata, raw_feature)
        elif all(attr in raw_feature for attr in keypoint_fingerprint):
            return KeypointAnnotation.deserialize_labelbox(raw_label_metadata, raw_feature)
        else:
            return Annotation(
                raw_metadata=raw_label_metadata,
                raw_annotation=raw_feature,
                type=AnnotationTypes.TYPE_UNKNOWN
            )


class AnnotationList(BaseModel):
    annotations: List[Annotation]

    @staticmethod
    def deserialize_sagemaker(raw_annotations, raw_metadata):
        annotations = []
        for idx, raw_annotation in enumerate(raw_annotations):
            annotations.append(Annotation.deserialize_sagemaker(raw_annotation, raw_metadata, idx))

        return AnnotationList(
            annotations=annotations
        )

    @staticmethod
    def deserialize_labelbox(raw_labels):
        annotations = []
        for raw_label_metadata in raw_labels:
            raw_features = json.loads(raw_label_metadata['label'])
            if 'objects' not in raw_features:
                continue

            for raw_feature in raw_features["objects"]:
                annotations.append(Annotation.deserialize_labelbox(raw_label_metadata, raw_feature))

        return AnnotationList(
            annotations=annotations
        )


class BoundingBoxAnnotation(Annotation):
    label: str
    width: float
    height: float
    top: float
    left: float

    @staticmethod
    def deserialize_sagemaker(raw_annotation, raw_metadata, idx):
        return BoundingBoxAnnotation(
            type=AnnotationTypes.TYPE_BOUNDING_BOX,
            label=raw_metadata["class-map"][str(raw_annotation["class_id"])],
            width=raw_annotation["width"],
            height=raw_annotation["height"],
            top=raw_annotation["top"],
            left=raw_annotation["left"],
            confidence=raw_metadata["objects"][idx]["confidence"],
            raw_annotation=raw_annotation,
            raw_metadata=raw_metadata,
            raw_metadata_annotation_idx=idx
        )

    @staticmethod
    def deserialize_labelbox(raw_label_metadata, raw_feature):
        return BoundingBoxAnnotation(
            type=AnnotationTypes.TYPE_BOUNDING_BOX,
            label=raw_feature["title"],
            width=raw_feature["width"],
            height=raw_feature["height"],
            top=raw_feature["top"],
            left=raw_feature["left"],
            raw_annotation=raw_feature,
            raw_metadata=raw_label_metadata
        )


class KeypointAnnotation(Annotation):
    label: str
    x: float
    y: float

    @staticmethod
    def deserialize_sagemaker(raw_annotation, raw_metadata, idx):
        pass

    @staticmethod
    def deserialize_labelbox(raw_label_metadata, raw_feature):
        return KeypointAnnotation(
            type=AnnotationTypes.TYPE_KEYPOINT,
            label=raw_feature["title"],
            x=raw_feature["point"]["x"],
            y=raw_feature["point"]["y"],
            raw_annotation=raw_feature,
            raw_metadata=raw_label_metadata
        )
