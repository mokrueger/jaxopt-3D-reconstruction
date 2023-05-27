from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageMetadata:
    identifier: str
    image_path: Optional[str]
    width: int
    height: int
