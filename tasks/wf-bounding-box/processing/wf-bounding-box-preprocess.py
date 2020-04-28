import json

# Event:
# {'version': '2018-10-16', 'labelingJobArn': 'test-labelingJob-arn', 'dataObject': {'source-ref': 's3://...jpg'}}
def lambda_handler(event, context):
    # Get source and source-ref if specified
    source = event['dataObject']['source'] if "source" in event['dataObject'] else None
    source_ref = event['dataObject']['source-ref'] if "source-ref" in event['dataObject'] else None

    metadata = event['dataObject']['metadata'] if "metadata" in event['dataObject'] else None


    # if source field present, take that otherwise take source-ref
    task_object = source if source is not None else source_ref

    # Build response object
    return {
        "taskInput": {
            "taskObject": task_object,
            "header": "Draw a box around each person",  # Required for sagemaker keypoint template
            "labels": ["person"]   # Required for sagemaker keypoint template
        }
    }
