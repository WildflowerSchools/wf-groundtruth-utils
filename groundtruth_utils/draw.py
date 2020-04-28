import boto3
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw

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
        img_pil = Image.open(bytes_stream, 'r').convert('RGB')
        img_draw = ImageDraw.Draw(img_pil, 'RGBA')
    finally:
        bytes_stream.close()

    # Load boxes
    for annotation in annotations:
        draw_shape_on_image(img_draw, annotation)

    return img_pil


def draw_annotations_and_save(image_uri, annotations, output_path, image_name):
    img_draw = draw_annotations(image_uri, annotations)
    img_draw.save("%s/%s" % (output_path, image_name), "PNG")


def draw_shape_on_image(img_draw, annotation):
    if annotation.type == AnnotationTypes.TYPE_BOUNDING_BOX:
        img_draw.rectangle([
            (int(annotation.left), int(annotation.top)),
            (int(annotation.left) + int(annotation.width), int(annotation.top) + int(annotation.height))
        ], fill=(0, 166, 156, 50), outline=(255, 255, 255))
