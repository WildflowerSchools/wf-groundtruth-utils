import click
import click_log
import json
import os

from .log import logger
from .core import fetch_annotations, fetch_jobs, generate_image_set, generate_manifest

click_log.basic_config(logger)


def validate_json(ctx, param, value):
    if value is None:
        return

    try:
        return json.loads(value)
    except ValueError:
        raise click.BadParameter("{0} need to be in JSON format".format(param.name))


@click.command(help="List groundtruth labeling jobs")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='sagemaker', help="platform to fetch from")
@click.option("-s", "--status", type=click.Choice(['inprogress', 'completed', 'failed',
                                                   'stopping', 'stopped']), default='completed', help="show jobs by status")
@click.option("-l", "--limit", type=int, default=None, help="limit number of jobs returned")
def list_jobs(platform, status, limit):
    jobs = fetch_jobs(platform=platform, status=status, limit=limit)
    click.echo(jobs.json(indent=2))


@click.command(help="List annotations for a given job")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='sagemaker', help="platform to fetch from")
@click.option('--no-consolidate', is_flag=False,
              help="default action is to consolidate multiple data labeler's annotations, use this flag to disable consolidation")
@click.argument("job_name")
def list_annotations(platform, no_consolidate, job_name):
    consolidate = not no_consolidate
    annotations = fetch_annotations(job_name, platform=platform, consolidate=consolidate)
    click.echo(annotations.json(indent=2))


@click.command(name="generate-image-set", help="Generate an annotated image set")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='sagemaker', help="platform to fetch from")
@click.option("-o", "--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, exports stored in '$OUTPUT/$job_name/$timestamp'")
@click.option("-m", "--mode", type=click.Choice(['combine', 'separate']), default='combine',
              help="'combine' - produce a single image containing all image annotations, 'separate' - produce a single image per image annotation")
@click.option('--no-consolidate', is_flag=False,
              help="default action is to consolidate multiple data labeler's annotations, use this flag to disable consolidation")
# TODO: Consider adding confidence filter
@click.argument("job_name")
def cli_generate_image_set(platform, output, mode, no_consolidate, job_name):
    consolidate = not no_consolidate
    generate_image_set(job_name, platform=platform, output=output, mode=mode, consolidate=consolidate)


@click.command(name="generate-manifest", help="Generate a job/dataset manifest file from an AWS folder")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='sagemaker', help="platform to fetch from")
@click.option("--metadata", type=str, callback=validate_json)
@click.argument("s3_images_uri")
def cli_generate_manifest(platform, metadata, s3_images_uri):
    manifest_raw = generate_manifest(s3_images_uri, platform=platform, metadata=metadata)
    click.echo(manifest_raw)


@click.command(name="generate-coco", help="Generate a coco formatted dataset from many jobs")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='sagemaker', help="platform to fetch from")
@click.argument("coco_generate_config")
def cli_generate_coco(platform, coco_generate_config):
    pass


@click_log.simple_verbosity_option(logger)
@click.group()
def cli():
    pass


cli.add_command(list_jobs)
cli.add_command(list_annotations)
cli.add_command(cli_generate_image_set)
cli.add_command(cli_generate_manifest)
