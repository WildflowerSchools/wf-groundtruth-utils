import click
import click_log
import json
import os

from .log import logger
from .core import fetch_annotations, fetch_jobs, generate_coco_dataset, generate_image_set, generate_mal_ndjson, generate_manifest, create_job
from .platforms.models.job import Job

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
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
@click.option("-s", "--status", type=click.Choice(['inprogress', 'completed', 'failed',
                                                   'stopping', 'stopped']), default='completed', help="show jobs by status")
@click.option("-l", "--limit", type=int, default=None, help="limit number of jobs returned")
@click.option('--raw', is_flag=False, help="print raw data from platform source")
def list_jobs(platform, status, limit, raw):
    jobs = fetch_jobs(platform=platform, status=status, limit=limit)
    output_args = {'indent': 2}
    if not raw:
        output_args['exclude'] = Job.exclude_raw()

    click.echo(jobs.json(**output_args))


@click.command(help="List annotations for a given job")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
@click.option('--raw', is_flag=False, help="print raw data from platform source")
@click.option('--no-consolidate', is_flag=False,
              help="default action is to consolidate multiple data labeler's annotations, use this flag to disable consolidation")
@click.argument("job_name")
def list_annotations(platform, no_consolidate, raw, job_name):
    consolidate = not no_consolidate
    annotations = fetch_annotations(job_name, platform=platform, consolidate=consolidate)
    output_args = {'indent': 2}
    if not raw:
        annotations.set_excluded_null()
        pass

    click.echo(annotations.json(**output_args))


@click.command(name="generate-image-set", help="Generate an annotated image set")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
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
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
@click.option("--metadata", type=str, callback=validate_json)
@click.argument("s3_images_uri")
def cli_generate_manifest(platform, metadata, s3_images_uri):
    manifest_raw = generate_manifest(s3_images_uri, platform=platform, metadata=metadata)
    click.echo(manifest_raw)


@click.command(name="generate-coco", help="Generate a coco formatted dataset from many jobs")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
@click.option("-o", "--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, exports stored in '$OUTPUT/coco-$timestamp.json'")
@click.argument("coco_generate_config", type=click.File('rb'))
def cli_generate_coco(platform, output, coco_generate_config):
    generate_coco_dataset(coco_generate_config, output, platform)


@click.command(name="create-job", help="Generate a labelbox job with an ontology and a dataset")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
@click.option("-o", "--ontology", type=click.File('rb'), required=True, help="Labelbox Ontology JSON")
@click.option("-d", "--dataset-id", type=str, required=True, help="Labelbox Dataset ID")
@click.argument("job_name")
def cli_create_job(platform, ontology, dataset_id, job_name):
    project = create_job(job_name, platform, ontology, dataset_id)
    click.echo(project.uid)


@click.command(name="generate-mal-ndjson", help="Generate a model assisted labeling NDJSON file for labelmaker")
@click.option("-o", "--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, exports stored in '$OUTPUT/labelmaker-mal-$timestamp.ndjson'")
@click.argument("job_name")
def cli_generate_mal_ndjson(output, job_name):
    generate_mal_ndjson(job_name, output, 'labelbox')


@click.command(name="upload-coco-labels-to-job", help="Load coco labels into a labelbox dataset")
@click.option("-c", "--coco-json", type=click.File('rb'), required=True, help="Coco formatted JSON data")
@click.argument("job_name")
def upload_coco_labels_to_job(coco_json, job_name):
    pass


@click_log.simple_verbosity_option(logger)
@click.group()
def cli():
    pass


cli.add_command(list_jobs)
cli.add_command(list_annotations)
cli.add_command(cli_generate_image_set)
cli.add_command(cli_generate_manifest)
cli.add_command(cli_generate_coco)
cli.add_command(cli_create_job)
