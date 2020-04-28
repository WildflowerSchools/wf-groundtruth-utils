import click
import os

from .core import fetch_annotations, fetch_jobs, generate_image_set


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
@click.argument("job_name")
def list_annotations(platform, job_name):
    annotations = fetch_annotations(job_name, platform=platform)
    click.echo(annotations.json(indent=2))


@click.command(name="generate-image-set", help="Generate an annotated image set")
@click.option("-p", "--platform", type=click.Choice(['sagemaker', 'labelbox'],
                                                    case_sensitive=False), default='sagemaker', help="platform to fetch from")
@click.option("-o", "--output", type=click.Path(), default="%s/output" % (os.getcwd()), help="output folder, exports stored in '$OUTPUT/$job_name/$timestamp'")
@click.option("-m", "--mode", type=click.Choice(['combine', 'separate']), default='combine', help="'combine' - produce a single image containing all image annotations, 'separate' - produce a single image per image annotation")
@click.argument("job_name")
def cli_generate_image_set(platform, output, mode, job_name):
    generate_image_set(job_name, platform=platform, output=output, mode=mode)


@click.group()
def cli():
    pass


cli.add_command(list_jobs)
cli.add_command(list_annotations)
cli.add_command(cli_generate_image_set)
