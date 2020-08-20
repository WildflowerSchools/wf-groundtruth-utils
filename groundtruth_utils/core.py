from datetime import datetime
import json
import ndjson
import pathlib
import time
import uuid

from pycocotools.coco import COCO as py_coco

from .aws.s3_util import upload_file_to_bucket
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


def generate_image_set(job_name='', platform='labelbox', output=os.getcwd(),
                       mode='combine', consolidate=True, naked=False,
                       filter_min_confidence=0.0, filter_min_labelers=3,
                       append_job_name=''):
    valid_modes = ['combine', 'separate']
    if mode.lower() not in valid_modes:
        raise Exception("'%s' invalid mode, must be combine|separate")

    active_platform = get_platform(platform)
    annotations, _ = active_platform.fetch_annotations(
        job_name,
        consolidate=consolidate,
        filter_min_confidence=filter_min_confidence,
        filter_min_labelers=filter_min_labelers)

    existing_image_names = []
    if append_job_name:
        append_job_images = active_platform.fetch_images(append_job_name).images
        existing_image_names = list(map(lambda img: img.external_id, append_job_images))

    now = datetime.now()
    instance_output_path = "%s/%s/%s" % (output, job_name, now.strftime("%m-%d-%YT%H:%M:%S"))

    pathlib.Path(instance_output_path).mkdir(parents=True, exist_ok=True)

    for image in annotations.images:
        image_name = os.path.basename(image.url)
        if mode == 'combine':
            image_annotations = image.annotations if not naked else []
            if image_name in existing_image_names:
                logger.info("Skipping '%s', image already in the 'append' dataset" % image_name)
                continue

            draw_annotations_and_save(image.url, image_annotations, instance_output_path, image_name)
        elif mode == 'separate':
            for idx, annotation in enumerate(image.annotations):
                image_annotation = [annotation] if not naked else []
                separated_file_name = get_separated_file_name(image_name, idx)
                if separated_file_name in existing_image_names:
                    logger.info("Skipping '%s', image already in the 'append' dataset" % separated_file_name)
                    continue

                draw_annotations_and_save(
                    image.url, image_annotation, instance_output_path, separated_file_name)


def generate_manifest(s3_images_uri, platform='labelbox', metadata=None):
    active_platform = get_platform(platform)
    return active_platform.generate_manifest(s3_images_uri, metadata=metadata)


def generate_coco_dataset(coco_generate_config, output=os.getcwd(), platform='labelbox', separate=False,
                          filter_min_confidence=0.0, filter_min_labelers=3,
                          validation_set=0.0, coco_file_name=None, validation_file_name=None):
    now = datetime.now()
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    generator = CocoGenerator()
    generator.load_data_from_platform(platform, coco_generate_config, separate,
                                      filter_min_confidence=filter_min_confidence,
                                      filter_min_labelers=filter_min_labelers)
    model = generator.model()

    output_file = "%s/%s" % (output, coco_file_name)
    if validation_set == 0.0:
        with open(output_file, "w") as f:
            f.write(model.json())
        logger.info("Saved coco dataset to %s" % output_file)
    else:
        val_output_file = "%s/%s" % (output, validation_file_name)
        [validation_model, train_model] = model.split(percent=validation_set)

        with open(output_file, "w") as f:
            f.write(train_model.json())
        logger.info("Saved coco training dataset to %s" % output_file)

        with open(val_output_file, "w") as f:
            f.write(validation_model.json())
        logger.info("Saved coco validation dataset to %s" % val_output_file)


def create_dataset(dataset_name='', manifest_file=None):
    manifest_json = json.load(manifest_file)

    platform = get_platform('labelbox')
    dataset = platform.create_dataset(dataset_name, manifest_json=manifest_json)
    return dataset


def create_job(job_name='', platform='labelbox', ontology_file=None, dataset_id=None):
    ontology_json = json.load(ontology_file)

    platform = get_platform(platform)
    job = platform.create_job(job_name=job_name, attrs={'ontology_json': ontology_json, 'dataset_id': dataset_id})
    return job


def upload_coco_labels_to_job(job_name='', coco_annotation_file=None):
    platform = get_platform('labelbox')
    platform.upload_coco_dataset(job_name, coco_annotation_file)


def generate_mal_ndjson(job_name='', output=os.getcwd(), coco_annotation_file=None, dataset_id=None):
    now = datetime.now()
    output_file = "%s/labelbox-mal-%s.ndjson" % (output, now.strftime("%m-%d-%YT%H:%M:%S"))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    platform = get_platform('labelbox')

    generator = CocoGenerator()
    if coco_annotation_file is not None:
        generator.load_data_from_pycoco(py_coco(coco_annotation_file))
    else:
        labelbox_image_urls = platform.fetch_images(job_name)
        generator.load_data_with_classifiers(labelbox_image_urls)

    model = generator.model()
    if len(model.annotations) == 0:
        logger.warn("No Machine Annotated Labels generated")
        return None

    logger.info("Generating ndjson records...")
    tic = time.time()
    mal_records = platform.generate_mal_ndjson(job_name, model, filter_dataset_id=dataset_id)
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

    return platform.create_mal_import_job(job_name, mal_public_file_url, deleteFeatures=False)


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
