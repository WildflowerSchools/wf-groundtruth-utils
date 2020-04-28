## Groundtruth Labeling Utilities

 - Interact with ground truth labeling jobs in Sagemaker and Labelbox.
 - Check job status and generate/visualize annotations.
 - Import as a module or use through CLI

### CLI

    just build
    groundtruth --help

### Development

Install Dev Packages

    just install-dev

Auto-format Code (PEP8 standard)

    just fmt
    
Need to debug the CLI?

    python -m groundtruth_utils ./groundtruth_utils/cli.py <<CMD>>
