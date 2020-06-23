from datetime import datetime
import json
import ndjson
import os
import pathlib
import requests
import tempfile
import time
import uuid

from .annotate import annotate_image
from .aws.s3_util import upload_file_to_bucket
from .coco.models.annotation import KeypointAnnotation as CocoKeypointAnnotation
from .coco.models.category import KeypointCategory as CocoKeypointCategory
from .coco.models.image import Image as CocoImage
from .coco_generator import CocoGenerator
from .draw import draw_annotations_and_save
from .helper import *
from .log import logger


def fetch_jobs(status='Completed', platform='labelbox', limit=None):
    active_platform = get_platform(platform)
    return active_platform.fetch_jobs(status, limit or 0)


def fetch_worker_annotations(job_name='', worker_ids=[]):
    print('Not Implemented')
    return


def fetch_annotations(job_name, platform='labelbox', consolidate=True):
    active_platform = get_platform(platform)
    return active_platform.fetch_annotations(job_name, consolidate)


def generate_image_set(job_name='', platform='labelbox', output=os.getcwd(), mode='combine', consolidate=True):
    valid_modes = ['combine', 'separate']
    if mode.lower() not in valid_modes:
        raise Exception("'%s' invalid mode, must be combine|separate")

    active_platform = get_platform(platform)
    annotations = active_platform.fetch_annotations(job_name, consolidate)

    now = datetime.now()
    instance_output_path = "%s/%s/%s" % (output, job_name, now.strftime("%m-%d-%YT%H:%M:%S"))

    pathlib.Path(instance_output_path).mkdir(parents=True, exist_ok=True)

    for image in annotations.images:
        image_name = os.path.basename(image.url)
        if mode == 'combine':
            draw_annotations_and_save(image.url, image.annotations, instance_output_path, image_name)
        elif mode == 'separate':
            for idx, annotation in enumerate(image.annotations):
                image_file_name = image_name.split(".", 1)
                draw_annotations_and_save(
                    image.url, [annotation], instance_output_path, "%s-%d.%s" %
                    (image_file_name[0], idx, image_file_name[1]))
    return


def generate_manifest(s3_images_uri, platform='labelbox', metadata=None):
    active_platform = get_platform(platform)
    return active_platform.generate_manifest(s3_images_uri, metadata=metadata)


def generate_coco_dataset(coco_generate_config, output=os.getcwd(), platform='labelbox'):
    now = datetime.now()
    output_file = "%s/coco-%s.json" % (output, now.strftime("%m-%d-%YT%H:%M:%S"))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    generator = CocoGenerator(coco_generate_config, platform)
    generator.load_data()
    model = generator.model()

    with open(output_file, "w") as f:
        f.write(model.json())

    logger.info("Saved coco dataset to %s" % output_file)


def create_job(job_name='', platform='labelbox', ontology_file=None, dataset_id=None):
    ontology_json = json.load(ontology_file)

    platform = get_platform(platform)
    job = platform.create_job(job_name=job_name, attrs={'ontology_json': ontology_json, 'dataset_id': dataset_id})
    return job


def upload_coco_labels_to_job(job_name='', coco_annotation_file=None):
    platform = get_platform('labelbox')
    platform.upload_coco_dataset(job_name, coco_annotation_file)


def generate_mal_ndjson(job_name='', output=os.getcwd()):
    now = datetime.now()
    output_file = "%s/labelbox-mal-%s.ndjson" % (output, now.strftime("%m-%d-%YT%H:%M:%S"))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    platform = get_platform('labelbox')
    labelbox_images = platform.fetch_images(job_name)

    final_annotations = []
    for image_idx, image in enumerate(labelbox_images):
        logger.info("Downloading image %s" % (image.url))
        tic = time.time()
        response = requests.get(image.url)
        if not response.ok:
            logger.warn("Failed downloading %s" % (image.url))
            continue

        logger.info('Done Downloading (t={:0.2f}s)'.format(time.time() - tic))

        with tempfile.NamedTemporaryFile() as temp_image:
            temp_image.write(response.content)
            temp_image.flush()

            logger.info("Annotating image %s" % (image.url))
            tic = time.time()
            annotations = annotate_image(temp_image.name)
            coco_annotations = []
            for annotation in annotations:
                coco_keypoint = CocoKeypointAnnotation(
                    image_id=image_idx,
                    bbox=annotation['bbox'],
                    category_id=CocoKeypointCategory.coco_person_category().id)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.NOSE,
                    annotation['keypoints'][0][0],
                    annotation['keypoints'][0][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_EYE,
                    annotation['keypoints'][1][0],
                    annotation['keypoints'][1][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_EYE,
                    annotation['keypoints'][2][0],
                    annotation['keypoints'][2][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_EAR,
                    annotation['keypoints'][3][0],
                    annotation['keypoints'][3][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_EAR,
                    annotation['keypoints'][4][0],
                    annotation['keypoints'][4][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_SHOULDER,
                    annotation['keypoints'][5][0],
                    annotation['keypoints'][5][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_SHOULDER,
                    annotation['keypoints'][6][0],
                    annotation['keypoints'][6][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_ELBOW,
                    annotation['keypoints'][7][0],
                    annotation['keypoints'][7][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_ELBOW,
                    annotation['keypoints'][8][0],
                    annotation['keypoints'][8][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_WRIST,
                    annotation['keypoints'][9][0],
                    annotation['keypoints'][9][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_WRIST,
                    annotation['keypoints'][10][0],
                    annotation['keypoints'][10][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_HIP,
                    annotation['keypoints'][11][0],
                    annotation['keypoints'][11][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_HIP,
                    annotation['keypoints'][12][0],
                    annotation['keypoints'][12][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_KNEE,
                    annotation['keypoints'][13][0],
                    annotation['keypoints'][13][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_KNEE,
                    annotation['keypoints'][14][0],
                    annotation['keypoints'][14][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.LEFT_ANKLE,
                    annotation['keypoints'][15][0],
                    annotation['keypoints'][15][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.RIGHT_ANKLE,
                    annotation['keypoints'][16][0],
                    annotation['keypoints'][16][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)
                coco_keypoint.add_keypoint(
                    CocoKeypointCategory.Keypoint.NECK,
                    annotation['keypoints'][17][0],
                    annotation['keypoints'][17][1],
                    CocoKeypointAnnotation.Visibility.VISIBILITY_LABELED_VISIBLE)

                coco_annotations.append(coco_keypoint)

            final_annotations.append({
                'image': CocoImage(id=image_idx, file_name=os.path.basename(image.url), width=0, height=0),
                'annotations': coco_annotations
            })
            logger.info('Done Annotating (t={:0.2f}s)'.format(time.time() - tic))

    if len(final_annotations) == 0:
        logger.warn("No Machine Annotated Labels generated")
        return None

    logger.info("Generating ndjson records...")
    tic = time.time()
    mal_records = platform.generate_mal_ndjson(job_name, final_annotations, deleteFeatures=True)
    logger.info('Done Generating ndjson records (t={:0.2f}s)'.format(time.time() - tic))

    if mal_records is None:
        logger.error('Error Generating ndjson records, record file returned None')
        return

    with open(output_file, 'w') as f:
        ndjson.dump(mal_records, f)

    logger.info("Saved labelbox model assisted label file to %s" % output_file)
    return output_file


def upload_mal_ndjson(job_name, mal_ndjson_file):
    bucket_name = os.environ.get('AWS_MAL_NDJSON_BUCKET', None)
    if bucket_name is None:
        logger.error("AWS_MAL_NDJSON_BUCKET required")
        return

    upload_path = os.environ.get('AWS_MAL_NDJSON_PATH', '').lstrip("/")

    platform = get_platform('labelbox')

    object_name = os.path.join(upload_path, '%s.ndjson' % (str(uuid.uuid4())))

    # Upload ndjson_file and get URL
    mal_public_file_url = upload_file_to_bucket(
        mal_ndjson_file, bucket_name, object_name, meta_data={
            'client_filename': os.path.basename(mal_ndjson_file)})

    if mal_public_file_url is None:
        logger.error("Failed to upload ndJSON file to AWS")
        return
    logger.info("Created ndJSON fileURL: %s" % mal_public_file_url)

    return platform.create_mal_import_job(job_name, mal_public_file_url)


def status_mal_ndjson(job_name, import_id):
    platform = get_platform('labelbox')
    return platform.get_status_mal_import_job(job_name, import_id)


def delete_mals(job_name, output, mal_ndjson_files):
    records = []
    for file in mal_ndjson_files:
        with open(file, 'r') as m:
            records.extend(ndjson.load(m))

    mal_records = list(map(lambda r: dict((k, r[k]) for k in ['uuid', 'schemaId', 'dataRow']), records))

    now = datetime.now()
    output_file = "%s/labelbox-mal-delete-%s.ndjson" % (output, now.strftime("%m-%d-%YT%H:%M:%S"))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    if mal_records is None or len(mal_records) == 0:
        logger.error('Error Generating ndjson delete records, record file returned None')
        return

    with open(output_file, 'w') as f:
        ndjson.dump(mal_records, f)

    upload_mal_ndjson(job_name, output_file)

    platform = get_platform('labelbox')
    platform.delete_unlabeled_features(job_name)
