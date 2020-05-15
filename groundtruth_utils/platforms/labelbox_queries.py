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
