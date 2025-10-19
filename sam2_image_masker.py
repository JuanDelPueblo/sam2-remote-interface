from PIL import Image
import torch
import numpy as np
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class SAM2ImageMasker:
    """
    Wrapper around SAM2 that loads the model/predictor and exposes a method
    to get masks for a given image.

    Parameters:
    - checkpoint: path to sam2 checkpoint (default same as original script)
    - model_cfg: model config path (default same as original script)
    - device: torch device or string ("cuda"/"mps"/"cpu"). If None, auto-detect.
    """

    def __init__(self, checkpoint="./checkpoints/sam2.1_hiera_large.pt", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", device=None):
        # select the device for computation (allow override via param)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # accept string or torch.device
            self.device = torch.device(device) if not isinstance(
                device, torch.device) else device

        # keep user-friendly print
        print(f"using device: {self.device}")

        if self.device.type == "cuda":
            # use bfloat16 for the entire session as in original script
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

        # build model and predictor
        self.sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def _prepare_image(self, image):
        """
        Accepts a PIL Image, numpy array (H,W,3) or a filepath string.
        Returns a numpy RGB array.
        """
        if isinstance(image, str):
            im = Image.open(image)
        elif isinstance(image, Image.Image):
            im = image
        elif isinstance(image, np.ndarray):
            # assume already RGB numpy array
            return image
        else:
            raise ValueError(
                "image must be a filepath, PIL.Image.Image or numpy array")

        return np.array(im.convert("RGB"))

    def set_image(self, image):
        """
        Accepts a PIL Image, numpy array (H,W,3) or a filepath string.
        Sets the image on the predictor.
        """
        image_np = self._prepare_image(image)
        self.predictor.set_image(image_np)

    def get_masks(self, point_coords=None, point_labels=None, input_boxes=None, multimask_output=True):
        """
        Return (masks, scores, logits) sorted by score desc.

        Parameters:
        - point_coords: numpy array of shape (N,2) or None (defaults to [[500,375]])
        - point_labels: numpy array of shape (N,) or None (defaults to [1])
        - multimask_output: bool

        Returns:
        (masks, scores, logits) where masks is a numpy array of shape (M, H, W)
        """
        if point_coords is None:
            point_coords = np.array([[500, 375]])
        if point_labels is None:
            point_labels = np.array([1])
        if input_boxes is None:
            input_boxes = np.array([[425, 600, 700, 875]])

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
