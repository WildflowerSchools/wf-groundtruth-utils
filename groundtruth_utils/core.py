import json
from datetime import datetime
import os
import pathlib

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


def generate_mal_ndjson(job_name='', output=os.getcwd(), platform='labelbox'):
    # Fetch Job's Ontology and create a way to map types:values to featureSchemaIds
    # Fetch Dataset
    # Combine all source annotations by external_ids of dest Job
    # Build a FeatureCollection using combined collections
    # - Map annotations to GeoJSON format (this will change in a week)
    # - Link features to related Ontology Schema ID
    # Optionally upload NDJSON file?
    pass


def upload_coco_labels_to_job(job_name='', coco_annotation_file=None):
    platform = get_platform('labelbox')
    platform.upload_coco_dataset(job_name, coco_annotation_file)
