from pydantic import BaseModel


class Image(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
