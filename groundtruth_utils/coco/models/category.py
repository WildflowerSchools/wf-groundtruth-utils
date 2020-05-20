from pydantic import BaseModel
from typing import List


class Category(BaseModel):
    id: int
    name: str
    supercategory: str


class KeypointCategory(Category):
    keypoints: List[str]
    skeleton: List[List[int]]

    @staticmethod
    def coco_person_category():
        return KeypointCategory(
            id=1,
            name="person",
            supercategory="person",
            keypoints=KeypointCategory.coco_person_keypoint_categories(),
            skeleton=KeypointCategory.coco_person_skeleton()
        )

    @staticmethod
    def coco_person_keypoint_categories():
        return [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    @staticmethod
    def coco_person_skeleton():
        return [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
            [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]
