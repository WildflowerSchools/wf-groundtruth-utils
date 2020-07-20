from pydantic import BaseModel


class Image(BaseModel):
    id: int
    file_name: str
    coco_url: str
    width: int
    height: int
