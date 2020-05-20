from enum import IntEnum
from functools import reduce

from pydantic import BaseModel
from typing import List

from groundtruth_utils.log import logger

from .category import KeypointCategory


class Annotation(BaseModel):
    id: int = 0
    bbox: List[int] = []
    image_id: int
    ignore: int = False
    area: int = 0
    iscrowd: int = False
    category_id: int

    def compute_area(self):
        if len(self.bbox) == 4:
            self.area = self.bbox[2] * self.bbox[3]


class KeypointAnnotation(Annotation):
    keypoints: List[int] = [0] * len(KeypointCategory.coco_person_keypoint_categories()) * 3
    num_keypoints: int = 0

    class Visibility(IntEnum):
        VISIBILITY_NOT_LABELED = 0
        VISIBILITY_LABELED_NOT_VISIBLE = 1
        VISIBILITY_LABELED_VISIBLE = 2

    def add_keypoint(self, category: str, x: int, y: int, visibility: Visibility):
        keypoint_catgories = KeypointCategory.coco_person_keypoint_categories()
        if category not in keypoint_catgories:
            logger.warn("keypoint category '%s' not found, not capturing keypoint" % category)
            return

        keypoint_index = keypoint_catgories.index(category) * 3
        self.keypoints[keypoint_index:keypoint_index + 2] = [x, y, visibility]

    def compute_num_keypoints(self):
        self.num_keypoints = reduce(lambda count, v: count + (v > 0), self.keypoints[2::3], 0)
