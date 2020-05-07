from pydantic import BaseModel
from typing import List


class AnnotationTypes:
    TYPE_BOUNDING_BOX = "BoundingBox"
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
    def deserialize_labelbox(raw_annotation):
        bounding_box_fingerprint = ['label', 'width', 'height', 'left', 'top']
        if all(attr in raw_annotation for attr in bounding_box_fingerprint):
            return BoundingBoxAnnotation.deserialize_labelbox(raw_annotation)
        else:
            return Annotation(
                raw_annotation=raw_annotation,
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
    def deserialize_labelbox(raw_annotations):
        annotations = []
        for idx, raw_annotation in enumerate(raw_annotations):
            annotations.append(Annotation.deserialize_labelbox(raw_annotation))

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
    def deserialize_labelbox(raw_annotation):
        return BoundingBoxAnnotation(
            type=AnnotationTypes.TYPE_BOUNDING_BOX,
            label=raw_annotation["label"],
            width=raw_annotation["width"],
            height=raw_annotation["height"],
            top=raw_annotation["top"],
            left=raw_annotation["left"],
            raw_annotation=raw_annotation
        )
