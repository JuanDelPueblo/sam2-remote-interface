import torch
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class SAM2ImageMasker:
    def __init__(self, checkpoint="./checkpoints/sam2.1_hiera_large.pt", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Utilizing device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # enable tf32 for new GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS."
            )

        np.random.seed(3)

        self.sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def set_image(self, image):
        self.predictor.set_image(image)

    def set_image_batch(self, images):
        self.predictor.set_image_batch(images)

    def predict(self, point_coords=None, point_labels=None, input_boxes=None, multimask_output=False):
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits

    def predict_batch(self, point_coords_batch=None, point_labels_batch=None, input_boxes_batch=None, multimask_output=False):
        masks_batch, scores_batch, logits_batch = self.predictor.predict_batch(
            point_coords_batch=point_coords_batch,
            point_labels_batch=point_labels_batch,
            box_batch=input_boxes_batch,
            multimask_output=multimask_output,
        )

        return masks_batch, scores_batch, logits_batch
