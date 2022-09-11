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

    def get_bounding_box(self):
        return self.bbox


class KeypointAnnotation(Annotation):
    keypoints: List[int] = [0] * len(KeypointCategory.coco_17_person_keypoint_categories()) * 3
    num_keypoints: int = 0

    class Visibility(IntEnum):
        VISIBILITY_NOT_LABELED = 0
        VISIBILITY_LABELED_NOT_VISIBLE = 1
        VISIBILITY_LABELED_VISIBLE = 2

    def get_keypoint_index(self, category: KeypointCategory):
        keypoint_categories = KeypointCategory.coco_17_person_keypoint_categories()
        if category.name not in keypoint_categories:
            logger.warn("keypoint category '%s' not found, not capturing keypoint" % category)
            return

        return keypoint_categories.index(category.name) * 3

    def add_keypoint(self, category: KeypointCategory, x: int, y: int, visibility: Visibility):
        keypoint_index = self.get_keypoint_index(category)
        self.keypoints[keypoint_index:keypoint_index + 3] = [x, y, visibility]

    def compute_num_keypoints(self):
        self.num_keypoints = reduce(lambda count, v: count + (v > 0), self.keypoints[2::3], 0)

    def get_keypoint_visibility(self, category: KeypointCategory):
        keypoint_index = self.get_keypoint_index(category)
        return self.__class__.Visibility(self.keypoints[keypoint_index + 2])

    def is_keypoint_visible(self, category: KeypointCategory):
        v = self.get_keypoint_visibility(category)
        return v == self.__class__.Visibility.VISIBILITY_LABELED_VISIBLE

    def is_keypoint_not_visible(self, category: KeypointCategory):
        v = self.get_keypoint_visibility(category)
        return v == self.__class__.Visibility.VISIBILITY_LABELED_NOT_VISIBLE

    def is_keypoint_visibility_labeled(self, category: KeypointCategory):
        v = self.get_keypoint_visibility(category)
        return v != self.__class__.Visibility.VISIBILITY_NOT_LABELED

    def get_keypoint_point(self, category: KeypointCategory):
        keypoint_index = self.get_keypoint_index(category)
        return self.keypoints[keypoint_index:keypoint_index + 2]
