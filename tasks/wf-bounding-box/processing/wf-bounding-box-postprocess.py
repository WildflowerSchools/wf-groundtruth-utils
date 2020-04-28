import json
import sys
from s3_helper import S3Client


def lambda_handler(event, context):
    """This is a sample Annotation Consolidation Lambda for custom labeling jobs. It takes all worker responses for the
    item to be labeled, and output a consolidated annotation.


    Parameters
    ----------
    event: dict, required
        Content of an example event

        {
            "version": "2018-10-16",
            "labelingJobArn": <labelingJobArn>,
            "labelCategories": [<string>],  # If you created labeling job using aws console, labelCategories will be null
            "labelAttributeName": <string>,
            "roleArn" : "string",
            "payload": {
                "s3Uri": <string>
            }
            "outputConfig":"s3://<consolidated_output configured for labeling job>"
         }


        Content of payload.s3Uri
        [
            {
                "datasetObjectId": <string>,
                "dataObject": {
                    "s3Uri": <string>,
                    "content": <string>
                },
                "annotations": [{
                    "workerId": <string>,
                    "annotationData": {
                        "content": <string>,
                        "s3Uri": <string>
                    }
               }]
            }
        ]

        As SageMaker product evolves, content of event object & payload.s3Uri will change. For a latest version refer following URL

        Event doc: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    consolidated_output: dict
        AnnotationConsolidation

        [
           {
                "datasetObjectId": <string>,
                "consolidatedAnnotation": {
                    "content": {
                        "<labelattributename>": {
                            # ... label content
                        }
                    }
                }
            }
        ]

        Return doc: https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3.html
    """
    consolidated_labels = []

    parsed_url = urlparse(event['payload']['s3Uri']);
    s3 = boto3.client('s3')
    textFile = s3.get_object(Bucket = parsed_url.netloc, Key = parsed_url.path[1:])
    filecont = textFile['Body'].read()
    annotations = json.loads(filecont);

    for dataset in annotations:
        for annotation in dataset['annotations']:
            new_annotation = json.loads(annotation['annotationData']['content'])
            label = {
                'datasetObjectId': dataset['datasetObjectId'],
                'consolidatedAnnotation' : {
                    'content': {
                        event['labelAttributeName']: {
                            'workerId': annotation['workerId'],
                            'result': new_annotation,
                            'labeledContent': dataset['dataObject']
                        }
                    }
                }
            }
            consolidated_labels.append(label)

    return consolidated_labels
