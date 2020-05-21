ALL_PROJECTS_METRICS_QUERY = """
query ProjectLabelingMetrics {
    projects(skip: %d, first: %d) {
      id
      name
      autoAuditNumberOfLabels
      autoAuditPercentage
      labelCount
      datasetSize: dataRowCount
      submitted: labelCount(where: { type: { name: "ANY" } })
      skipped: labelCount(where: { type: { name: "SKIP" } })
      createdAt
      updatedAt
      __typename
    }
}
"""

ALL_ANNOTATIONS_QUERY = """
query GetAllAnnotations($id: ID!){
  project(where:{id: $id}){
    dataRows(skip: %d, first: %d) {
      id
      externalId
      rowData
      labels(first: 100, where: {
        project: {
            id: $id
        }
       }) {
        id
        createdBy{
          name
          email
        }
        secondsToLabel
        agreement
        label
      }
    }
  }
}
"""

GET_IMAGE_LABELING_FRONTEND_ID = """
query GetImageLabelingInterfaceId {
  labelingFrontends(where:{
    iframeUrlPath:"https://editor.labelbox.com"
  }){
    id
  }
}
"""

CONFIGURE_INTERFACE_FOR_PROJECT = """
mutation ConfigureInterfaceFromAPI($projectId: ID!, $customizationOptions: String!, $labelingFrontendId: ID!, $organizationId: ID!) {
    createLabelingFrontendOptions(data:{
      customizationOptions: $customizationOptions,
      project:{
        connect:{
          id: $projectId
        }
      }
      labelingFrontend:{
        connect:{
          id:$labelingFrontendId
        }
      }
      organization:{
        connect:{
          id: $organizationId
        }
      }
    }){
      id
    }
}
"""

ATTACH_DATASET_AND_FRONTEND = """
mutation attach_dataset_and_frontend($projectId: ID!, $datasetId: ID!, $labelingFrontendId: ID!, $date: DateTime!){
  updateProject(
    where:{
      id:$projectId
    },
    data:{
      setupComplete: $date,
      datasets:{
        connect:{
          id:$datasetId
        }
      },
      labelingFrontend:{
        connect:{
          id:$labelingFrontendId
        }
      }
    }
  ){
    id
  }
}
"""

DELETE_PROJECT = """
mutation delete_project($projectId: ID!) {
  deleteProject(project: {id: "ckagyny9gnwka0714utcfxrer"}) {
    id
  }
}
"""