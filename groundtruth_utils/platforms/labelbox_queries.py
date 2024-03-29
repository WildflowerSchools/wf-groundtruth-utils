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
query GetAllAnnotations($projectId: ID!){
  project(where:{id: $projectId}){
    dataRows(skip: %d, first: %d) {
      id
      externalId
      rowData
      labels {
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

ALL_PROJECT_IMAGES_QUERY = """
query AllProjectImages($projectId: ID!) {
    project (where:
        {id: $projectId}
    ) {
    dataRows(skip: %d, first: %d) {
      id
      externalId
      rowData
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
  deleteProject(project: {id: $projectId}) {
    id
  }
}
"""

GET_PROJECT_ONTOLOGY = """
query GetProjectOntology($projectId: ID!) {
    project (where:
        {id: $projectId}
    ) {
        name
        ontology {
            normalized
        }
    }
}
"""

_FEATURE_CACHE_FIELDS_FRAGMENT = """
fragment FeatureCacheFields on Feature {
  __typename
  id
  updatedAt
  deleted
  dataRow {
    __typename
    id
  }
}
"""

_CREATE_NEW_OBJECT_FEATURE_MUTATION = """
mutation CreateNewObjectFeature($id: ID, $schemaId: ID!, $projectId: ID!, $dataRowId: ID!, $content: Json!, $label: WhereUniqueIdInput, $secondsSpent: Float!) {
  createObjectFeature(data: {id: $id, schema: {id: $schemaId}, project: {id: $projectId}, dataRow: {id: $dataRowId}, label: $label, content: $content, secondsSpent: $secondsSpent}) {
    id
    result {
      ...FeatureCacheFields
      label {
        id
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

CREATE_NEW_OBJECT_FEATURE = """
{create_new_object_feature_mutation}
{feature_cache_fields}
""".format(create_new_object_feature_mutation=_CREATE_NEW_OBJECT_FEATURE_MUTATION, feature_cache_fields=_FEATURE_CACHE_FIELDS_FRAGMENT)

_CREATE_NEW_NESTED_CLASSIFICATION_FEATURE_MUTATION = """
mutation CreateNewNestedClassificationFeature($questionSchemaId: ID!, $optionSchemaIds: [ID!]!, $secondsSpent: Float!, $parentFeatureId: ID!) {
  objectFeature(feature: {id: $parentFeatureId}) {
    createClassificationFeature(data: {schema: {id: $questionSchemaId}, secondsSpent: $secondsSpent}) {
      setOptions(schemaIds: $optionSchemaIds, additionalSecondsSpent: 0) {
        result {
          ...FeatureCacheFields
          __typename
        }
        descendants {
          ...FeatureCacheFields
          schema {
            id
            __typename
          }
          parent {
            id
            __typename
          }
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

CREATE_NEW_NESTED_CLASSIFICATION_FEATURE = """
{create_new_nested_classification_mutation}
{feature_cache_fields}
""".format(create_new_nested_classification_mutation=_CREATE_NEW_NESTED_CLASSIFICATION_FEATURE_MUTATION, feature_cache_fields=_FEATURE_CACHE_FIELDS_FRAGMENT)

_UPDATE_CLASSIFICATION_OPTIONS_MUTATION = """
mutation UpdateClassificationOptions($questionFeatureId: ID!, $optionSchemaIds: [ID!]!, $additionalSecondsSpent: Float!) {
  classificationFeature(feature: {id: $questionFeatureId}) {
    setOptions(schemaIds: $optionSchemaIds, additionalSecondsSpent: $additionalSecondsSpent) {
      result {
        ...FeatureCacheFields
        __typename
      }
      descendants {
        ...FeatureCacheFields
        schema {
          id
          __typename
        }
        parent {
          id
          __typename
        }
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

UPDATE_CLASSIFICATION_OPTIONS = """
{update_classification_options_mutation}
{feature_cache_fields}
""".format(update_classification_options_mutation=_UPDATE_CLASSIFICATION_OPTIONS_MUTATION, feature_cache_fields=_FEATURE_CACHE_FIELDS_FRAGMENT)

CREATE_LABEL_FROM_FEATURES = """
mutation CreateLabelFromFeatures($projectId: ID!, $dataRowId: ID!, $featureIds: [ID!]!, $secondsSpent: Float!, $templateId: ID) {
  createLabelFromFeatures(data: {featureIds: $featureIds, project: {id: $projectId}, secondsSpent: $secondsSpent, dataRow: {id: $dataRowId}, templateId: $templateId}) {
    id
    label
    updatedAt
    __typename
  }
}
"""

_ALL_FEATURES_FOR_DATAROW_QUERY = """
query GetExistingFeatures($projectId: ID!, $dataRowId: ID!) {
  project(where: {id: $projectId}) {
    id
    dataRows{
      id
    }
    featuresForDataRow(where: {dataRow: {id: $dataRowId}}, skip: %d, first: %d) {
      ...FeatureCacheFields
      content
      createdAt
      createdBy {
        id
        name
      }
      schema {
        id
        kind
        definition
        __typename
      }
      parent {
        id
        __typename
      }
      children {
        id
        __typename
      }
      label {
        id
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

ALL_FEATURES_FOR_DATAROW_QUERY = """
{all_features_for_datarow_query}
{feature_cache_fields}
""".format(all_features_for_datarow_query=_ALL_FEATURES_FOR_DATAROW_QUERY, feature_cache_fields=_FEATURE_CACHE_FIELDS_FRAGMENT)

DELETE_FEATURE = """
mutation DeleteFeature($featureId: ID!) {
    deleteFeature(feature: {id: $featureId}) {
        id
        deleted
        project {
            id
            name
        }
        dataRow {
            id
        }
    }
}
"""

CREATE_MAL_IMPORT_REQUEST = """
mutation CreateMALImportRequest($projectId: ID!, $importName: String!, $fileUrl: String!) {
    createBulkImportRequest(data: {
            projectId: $projectId,
            name: $importName,
            fileUrl: $fileUrl}) {
        id
    }
}
"""

GET_STATUS_MAL_IMPORT_REQUEST = """
query StatusMALImportRequest($projectId: ID!, $importName: String!) {
    bulkImportRequest(where: {
            projectId: $projectId,
            name: $importName}) {
        id
        name
        state
        statusFileUrl
        errorFileUrl
    }
}
"""
