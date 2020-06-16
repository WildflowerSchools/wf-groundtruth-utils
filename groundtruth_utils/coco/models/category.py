from enum import IntEnum, Enum, auto

from pydantic import BaseModel
from typing import List


class Category(BaseModel):
    id: int
    name: str
    supercategory: str


class KeypointCategory(Category):
    keypoints: List[str]
    skeleton: List[List[int]]

    class Keypoint(Enum):
        def __new__(cls, *args):
            if len(args) == 2:
                if isinstance(args, tuple):
                    obj = object.__new__(cls)
                    obj._value_ = args
                    return obj

        @property
        def id(self):
            return self.value[0]

        @property
        def name(self):
            return self.value[1]

        NOSE = 1, 'nose'
        LEFT_EYE = 2, 'left_eye'
        RIGHT_EYE = 3, 'right_eye'
        LEFT_EAR = 4, 'left_ear'
        RIGHT_EAR = 5, 'right_ear'
        LEFT_SHOULDER = 6, 'left_shoulder'
        RIGHT_SHOULDER = 7, 'right_shoulder'
        LEFT_ELBOW = 8, 'left_elbow'
        RIGHT_ELBOW = 9, 'right_elbow'
        LEFT_WRIST = 10, 'left_wrist'
        RIGHT_WRIST = 11, 'right_wrist'
        LEFT_HIP = 12, 'left_hip'
        RIGHT_HIP = 13, 'right_hip'
        LEFT_KNEE = 14, 'left_knee'
        RIGHT_KNEE = 15, 'right_knee'
        LEFT_ANKLE = 16, 'left_ankle'
        RIGHT_ANKLE = 17, 'right_ankle'
        NECK = 18, 'neck'

    # Python Enum instantiation will overwrite the Class "__new__" function
    # This function and the setattr below manually rewrite "__new__" once
    # more after Python
    def KeypointLookup(cls, value):
        keys = cls.__members__.keys()
        if isinstance(value, str):
            key = value.replace(' ', '_').upper()
            if key in keys:
                return getattr(cls, key)
        elif isinstance(value, int) and 0 < value <= len(keys):
            return list(cls.__members__.values())[value - 1]
        else:
            return None

    setattr(Keypoint, '__new__', KeypointLookup)

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
            KeypointCategory.Keypoint.NOSE.name,
            KeypointCategory.Keypoint.LEFT_EYE.name,
            KeypointCategory.Keypoint.RIGHT_EYE.name,
            KeypointCategory.Keypoint.LEFT_EAR.name,
            KeypointCategory.Keypoint.RIGHT_EAR.name,
            KeypointCategory.Keypoint.LEFT_SHOULDER.name,
            KeypointCategory.Keypoint.RIGHT_SHOULDER.name,
            KeypointCategory.Keypoint.LEFT_ELBOW.name,
            KeypointCategory.Keypoint.RIGHT_ELBOW.name,
            KeypointCategory.Keypoint.LEFT_WRIST.name,
            KeypointCategory.Keypoint.RIGHT_WRIST.name,
            KeypointCategory.Keypoint.LEFT_HIP.name,
            KeypointCategory.Keypoint.RIGHT_HIP.name,
            KeypointCategory.Keypoint.LEFT_KNEE.name,
            KeypointCategory.Keypoint.RIGHT_KNEE.name,
            KeypointCategory.Keypoint.LEFT_ANKLE.name,
            KeypointCategory.Keypoint.RIGHT_ANKLE.name,
            KeypointCategory.Keypoint.NECK.name
        ]

    @staticmethod
    def coco_person_skeleton():
        return [
            [KeypointCategory.Keypoint.LEFT_ANKLE.id, KeypointCategory.Keypoint.LEFT_KNEE.id],
            [KeypointCategory.Keypoint.LEFT_KNEE.id, KeypointCategory.Keypoint.LEFT_HIP.id],
            [KeypointCategory.Keypoint.RIGHT_ANKLE.id, KeypointCategory.Keypoint.RIGHT_KNEE.id],
            [KeypointCategory.Keypoint.RIGHT_KNEE.id, KeypointCategory.Keypoint.RIGHT_HIP.id],
            [KeypointCategory.Keypoint.LEFT_HIP.id, KeypointCategory.Keypoint.RIGHT_HIP.id],
            [KeypointCategory.Keypoint.LEFT_SHOULDER.id, KeypointCategory.Keypoint.LEFT_HIP.id],
            [KeypointCategory.Keypoint.RIGHT_SHOULDER.id, KeypointCategory.Keypoint.RIGHT_HIP.id],
            [KeypointCategory.Keypoint.LEFT_SHOULDER.id, KeypointCategory.Keypoint.RIGHT_SHOULDER.id],
            [KeypointCategory.Keypoint.LEFT_SHOULDER.id, KeypointCategory.Keypoint.LEFT_ELBOW.id],
            [KeypointCategory.Keypoint.RIGHT_SHOULDER.id, KeypointCategory.Keypoint.RIGHT_ELBOW.id],
            [KeypointCategory.Keypoint.LEFT_ELBOW.id, KeypointCategory.Keypoint.LEFT_WRIST.id],
            [KeypointCategory.Keypoint.RIGHT_ELBOW.id, KeypointCategory.Keypoint.RIGHT_WRIST.id],
            [KeypointCategory.Keypoint.LEFT_EYE.id, KeypointCategory.Keypoint.RIGHT_EYE.id],
            [KeypointCategory.Keypoint.NOSE.id, KeypointCategory.Keypoint.LEFT_EYE.id],
            [KeypointCategory.Keypoint.NOSE.id, KeypointCategory.Keypoint.RIGHT_EYE.id],
            [KeypointCategory.Keypoint.LEFT_EYE.id, KeypointCategory.Keypoint.LEFT_EAR.id],
            [KeypointCategory.Keypoint.RIGHT_EYE.id, KeypointCategory.Keypoint.RIGHT_EAR.id],
            [KeypointCategory.Keypoint.LEFT_EAR.id, KeypointCategory.Keypoint.LEFT_SHOULDER.id],
            [KeypointCategory.Keypoint.RIGHT_EAR.id, KeypointCategory.Keypoint.RIGHT_SHOULDER.id],
            [KeypointCategory.Keypoint.NOSE.id, KeypointCategory.Keypoint.NECK.id],
            [KeypointCategory.Keypoint.NECK.id, KeypointCategory.Keypoint.LEFT_SHOULDER.id],
            [KeypointCategory.Keypoint.NECK.id, KeypointCategory.Keypoint.RIGHT_SHOULDER.id]
        ]
