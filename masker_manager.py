from enum import Enum
from typing import Optional

import sam2_image_masker as sim
import sam2_video_masker as svm


class MaskerType(str, Enum):
    image = "image"
    video = "video"


class MaskerManager:
    def __init__(self):
        self.masker = None
        self.active_masker_type: Optional[MaskerType] = None

    def set_masker_type(self, masker_type: MaskerType):
        if self.active_masker_type == masker_type:
            return

        if masker_type == MaskerType.image:
            self.masker = sim.SAM2ImageMasker()
        elif masker_type == MaskerType.video:
            self.masker = svm.SAM2VideoMasker()
        else:
            self.masker = None
            self.active_masker_type = None
            raise ValueError("Invalid masker type")

        self.active_masker_type = masker_type

    def get_masker(self):
        return self.masker

    def get_active_masker_type(self):
        return self.active_masker_type


masker_manager = MaskerManager()
