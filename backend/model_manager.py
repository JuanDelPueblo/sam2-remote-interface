from enum import Enum
from typing import Optional

import sam2_image_masker as sim
import sam2_video_masker as svm
import co_tracker as cot


class ModelType(str, Enum):
    image = "image"
    video = "video"
    tracking = "tracking"


class ModelManager:
    def __init__(self):
        self.model = None
        self.active_model_type: Optional[ModelType] = None

    def set_model_type(self, model_type: ModelType):
        if self.active_model_type == model_type:
            return

        if model_type == ModelType.image:
            self.model = sim.SAM2ImageMasker()
        elif model_type == ModelType.video:
            self.model = svm.SAM2VideoMasker()
        elif model_type == ModelType.tracking:
            self.model = cot.CoTracker()
        else:
            self.model = None
            self.active_model_type = None
            raise ValueError("Invalid model type")

        self.active_model_type = model_type

    def get_model(self):
        return self.model

    def get_active_model_type(self):
        return self.active_model_type


model_manager = ModelManager()
