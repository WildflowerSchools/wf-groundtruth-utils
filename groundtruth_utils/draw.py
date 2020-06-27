import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw

from .aws.s3_util import download_fileobj_as_bytestream
from .log import logger
from .platforms.models.annotation import AnnotationTypes


def get_s3_image_as_pil(image_uri):
    try:
        s3_client = boto3.client('s3')
        bytes_stream = download_fileobj_as_bytestream(s3_client, image_uri)
    except ClientError as e:
        print("Unexpected error fetching %s: %s" % (image_uri, e))
        raise e

    # Load image
    try:
        bytes_stream.seek(0)
        img_pil = Image.open(bytes_stream, 'r').convert('RGB')
    finally:
        bytes_stream.close()

    return img_pil


def draw_annotations(image_uri, annotations):
    img_pil = get_s3_image_as_pil(image_uri)
    img_draw = ImageDraw.Draw(img_pil, 'RGBA')

    # Load boxes
    for annotation in annotations:
        draw_shape_on_image(img_draw, annotation)

    return img_pil


def draw_annotations_and_save(image_uri, annotations, output_path, image_name):
    img_draw = draw_annotations(image_uri, annotations)
    img_draw.save("%s/%s" % (output_path, image_name), "PNG")
    logger.info("Saved image %s/%s (%d annotations)" % (output_path, image_name, len(annotations)))


def draw_shape_on_image(img_draw, annotation):
    if annotation.type == AnnotationTypes.TYPE_BOUNDING_BOX:
        img_draw.rectangle([
            (int(annotation.left), int(annotation.top)),
            (int(annotation.left) + int(annotation.width), int(annotation.top) + int(annotation.height))
        ], fill=(0, 166, 156, 50), outline=(255, 255, 255), width=3)
    elif annotation.type == AnnotationTypes.TYPE_KEYPOINT:
        radius = 3
        img_draw.ellipse([
            (int(annotation.x) - radius, int(annotation.y) - radius),
            (int(annotation.x) + radius, int(annotation.y) + radius)
        ], fill=(65, 255, 126, 95), outline=(65, 255, 126))
