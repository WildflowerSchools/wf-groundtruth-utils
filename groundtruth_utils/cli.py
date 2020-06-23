import click
import click_log
import json
import os

from .annotate import annotate_image
from .log import logger
from .core import create_job, delete_mals, fetch_annotations, fetch_jobs, generate_coco_dataset, generate_image_set, generate_mal_ndjson, generate_manifest, upload_coco_labels_to_job, upload_mal_ndjson, status_mal_ndjson
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


@click.command(name="create-job", help="Generate a Labelbox job with an ontology and a dataset")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='labelbox', help="platform to fetch from")
@click.option("-o", "--ontology", type=click.File('rb'), required=True, help="Labelbox Ontology JSON")
@click.option("-d", "--dataset-id", type=str, required=True, help="Labelbox Dataset ID")
@click.argument("job_name")
def cli_create_job(platform, ontology, dataset_id, job_name):
    project = create_job(job_name, platform, ontology, dataset_id)
    click.echo(project.uid)


@click.command(name="upload-coco-labels-to-job", help="Load coco labels into a Labelbox dataset")
@click.option("-c", "--coco-json", type=click.Path(exists=True), required=True, help="Coco formatted JSON data")
@click.argument("job_name")
def cli_upload_coco_labels_to_job(coco_json, job_name):
    upload_coco_labels_to_job(job_name, coco_json)


@click.command(name="generate-mal-ndjson", help="Generate a model assisted labeling ndJSON file for Labelbox")
@click.option("-o", "--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, exports stored in '$OUTPUT/labelmaker-mal-$timestamp.ndjson'")
@click.option('--upload', is_flag=True, default=False, help="Upload the ndJSON file after it's generated")
@click.argument("job_name")
@click.pass_context
def cli_generate_mal_ndjson(ctx, output, job_name, upload):
    file_name = generate_mal_ndjson(job_name, output)

    if upload and file_name:
        ctx.invoke(cli_upload_mal_ndjson, mal_file=file_name, job_name=job_name)


@click.command(name="upload-mal-ndjson", help="Upload a MAL ndJSON file to a labelbox job")
@click.option("-m", "--mal-file", type=click.Path(exists=True), required=True,
              help="ndJSON formatted file for Labelbox MAL annotations")
@click.argument("job_name")
def cli_upload_mal_ndjson(mal_file, job_name):
    import_id = upload_mal_ndjson(job_name, mal_file)
    if import_id is not None:
        click.echo("Started MAL import job: %s" % (import_id))


@click.command(name="status-mal-ndjson", help="Check the status of a MAL ndJSON import job")
@click.option("-i", "--import-id", type=str, required=True, help="ndJSON import ID")
@click.argument("job_name")
def cli_status_mal_ndjson(import_id, job_name):
    status = status_mal_ndjson(job_name, import_id)
    if status is not None:
        click.echo(json.dumps(status, indent=4))


@click.command(name="delete-mals", help="Delete Model Assisted Label")
@click.option("-o", "--output", type=click.Path(), default="%s/output" % (os.getcwd()),
              help="output folder, ndjson delete file stored in '$OUTPUT/labelmaker-mal-delete-$timestamp.ndjson'")
@click.option("-m", "--mal-files", type=click.Path(exists=True), required=True, multiple=True,
              help="ndJSON formatted file(s) for Labelbox MAL annotations, multiple allowed")
@click.argument("job_name")
def cli_delete_mals(mal_files, output, job_name):
    delete_mals(job_name, output, mal_files)


@click.command(name="annotate-image", help="Annotate an image")
@click.option("-i", "--image", type=click.Path(exists=True), required=True, help="Image to annotate")
def cli_annotate_image(image):
    result = annotate_image(image)
    click.echo(json.dumps(result, indent=4))


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
cli.add_command(cli_upload_coco_labels_to_job)
cli.add_command(cli_generate_mal_ndjson)
cli.add_command(cli_upload_mal_ndjson)
cli.add_command(cli_status_mal_ndjson)
cli.add_command(cli_delete_mals)
cli.add_command(cli_annotate_image)
