import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import os

from .aws.s3_util import download_fileobj_as_bytestream
from .platforms.models.annotation import AnnotationTypes


def draw_annotations(image_uri, annotations):
    try:
        s3_client = boto3.client('s3')
        bytes_stream = download_fileobj_as_bytestream(s3_client, image_uri)
    except ClientError as e:
        print("Unexpected error fetching %s: %s" % (image_uri, e))
        raise e

    # Load image
    try:
        bytes_stream.seek(0)
        img_str = bytes_stream.read()
        np_arr = np.fromstring(img_str, dtype=np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    finally:
        bytes_stream.close()

    # Load boxes
    for annotation in annotations:
        draw_shape_on_image(img_np, annotation)

    return img_np


def draw_annotations_and_save(image_uri, annotations, output_path, image_name):
    img_np = draw_annotations(image_uri, annotations)
    cv2.imwrite("%s/%s" % (output_path, image_name), img_np)


def draw_shape_on_image(img_np, annotation):
    if annotation.type == AnnotationTypes.TYPE_BOUNDING_BOX:
        # image = np.array(img_np)
        cv2.rectangle(img_np, (int(annotation.left), int(annotation.top)), (int(annotation.left) +
                                                                            int(annotation.width), int(annotation.top) + int(annotation.height)), (0, 255, 0), 2)
