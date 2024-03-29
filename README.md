## Groundtruth Labeling Utilities

 - Interact with ground truth labeling jobs in Sagemaker and Labelbox.
 - Check job status and generate/visualize annotations.
 - Import as a module or use through CLI

### Environment

Copy `.env.template` to `.env` and fill in required ENV vars. `.env` file will automatically be loaded when module executed from root folder. 

### CLI

    just install
    groundtruth --help
    
*Generate manifest*

    # Create a manifest for Sagemaker with custom labels metadata, generates JSONL
    groundtruth generate-manifest --metadata='{"labels": ["left-shoulder", "right-shoulder"]}' <<S3 URI TO IMAGE FOLDER>> > output/sagemaker-dataset.manifest
    
    # Create a manifest for Labelbox (metadata is ignored for Labelbox), generates JSON
    groundtruth generate-manifest -p labelbox <<S3 URI TO IMAGE FOLDER>> > output/labelbox-dataset.json

*Generate COCO dataset from keypoints*

    groundtruth generate-coco -m combine --filter-min-labelers 2 --coco-file-name wf.train.json ./groundtruth_utils/config/generate_coco_from_keypoint_labels.yml

### Development

Install Dev Packages

    just install

Auto-format Code (PEP8 standard)

    just fmt
    
Need to debug the CLI?

    python -m groundtruth_utils ./groundtruth_utils/cli.py <<CMD>>
