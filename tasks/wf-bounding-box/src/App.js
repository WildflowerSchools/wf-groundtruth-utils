import React from 'react'
import ReactImageAnnotate from "react-image-annotate"

import './App.css';

const App = () => (
  <ReactImageAnnotate
    selectedImage="https://wildflower-tech-public.s3.us-east-2.amazonaws.com/sagemaker-ui/000000000308.jpg"
    taskDescription="Draw a bounding box around every person"
    images={[{ src: "https://wildflower-tech-public.s3.us-east-2.amazonaws.com/sagemaker-ui/000000000308.jpg", name: 'test'}]}
    regionClsList={["Person", "Group"]}
    enabledTools={["select", "create-box"]}
  />
)

export default App
