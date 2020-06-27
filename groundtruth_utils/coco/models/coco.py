from pydantic import BaseModel
from typing import List

from .annotation import Annotation, KeypointAnnotation
from .category import BaseCategory
from .image import Image


class Coco(BaseModel):
    images: List[Image] = []
    annotations: List[Annotation] = []
    categories: List[BaseCategory] = []

    def get_annotations_for_image(self, image_id):
        return list(filter(lambda a: a.image_id == image_id, self.annotations))

    def load_pycoco(self, pycoco):
        for image in pycoco.loadImgs(pycoco.getImgIds()):
            self.images.append(Image(
                id=image['id'],
                file_name=image['file_name'],
                width=image['width'],
                height=image['height']
            ))
            annotation_ids = pycoco.getAnnIds(imgIds=[image['id']])
            for annotation in pycoco.loadAnns(annotation_ids):
                self.annotations.append(KeypointAnnotation(
                    id=annotation['id'],
                    image_id=image['id'],
                    bbox=annotation['bbox'],
                    ignore=annotation['ignore'],
                    area=annotation['area'],
                    iscrowd=annotation['iscrowd'],
                    category_id=annotation['category_id'],
                    keypoints=annotation['keypoints'],
                    num_keypoints=annotation['num_keypoints']
                ))
