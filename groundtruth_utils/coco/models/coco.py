from pydantic import BaseModel
from typing import List

from .annotation import Annotation
from .category import Category
from .image import Image


class Coco(BaseModel):
    images: List[Image] = []
    annotations: List[Annotation] = []
    categories: List[Category] = []
