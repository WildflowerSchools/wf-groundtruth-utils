from datetime import datetime
import os
import pathlib

from .draw import draw_annotations_and_save
from .helper import *


def fetch_jobs(status='Completed', platform='sagemaker', limit=None):
    active_platform = get_platform(platform)
    return active_platform.fetch_jobs(status, limit or 0)


def fetch_worker_annotations(job_name='', worker_ids=[]):
    print('Not Implemented')
    return


def fetch_annotations(job_name, platform='sagemaker'):
    active_platform = get_platform(platform)
    return active_platform.fetch_annotations(job_name)


def generate_image_set(job_name='', platform='sagemaker', output=os.getcwd(), mode='combine'):
    valid_modes = ['combine', 'separate']
    if mode.lower() not in valid_modes:
        raise Exception("'%s' invalid mode, must be combine|separate")

    active_platform = get_platform(platform)
    annotations = active_platform.fetch_annotations(job_name)

    now = datetime.now()
    instance_output_path = "%s/%s/%s" % (output, job_name, now.strftime("%d-%m-%YT%H:%M:%S"))
    pathlib.Path(instance_output_path).mkdir(parents=True, exist_ok=True)

    for image in annotations.images:
        image_name = os.path.basename(image.url)
        if mode == 'combine':
            draw_annotations_and_save(image.url, image.annotations, instance_output_path, image_name)
        elif mode == 'separate':
            for idx, annotation in enumerate(image.annotations):
                image_file_name = image_name.split(".", 1)
                draw_annotations_and_save(image.url, [annotation], instance_output_path, "%s-%d.%s" % (image_file_name[0], idx, image_file_name[1]))
    return
