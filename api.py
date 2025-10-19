from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import sam2_image_masker as sim
from utils import *

app = FastAPI()

masker = sim.SAM2ImageMasker()
image = None  # Global variable to store the current image


class Points(BaseModel):
    point_coords: Optional[list[list[float]]] = None
    point_labels: Optional[list[int]] = None
    multimask_output: bool = True


@app.get("/")
async def root():
    return {"message": "SAM 2 Image Masker API"}


@app.post("/set_image")
async def set_image(request: UploadFile = File(...)):
    global image
    image = Image.open(request.file)
    image = np.array(image.convert("RGB"))
    masker.set_image(image)
    return {"message": "Image set successfully"}


@app.post("/get_masks")
async def get_masks(points: Points):
    point_coords = None
    point_labels = None
    multimask_output = points.multimask_output

    if points.point_coords is not None:
        point_coords = np.array(points.point_coords)
    if points.point_labels is not None:
        point_labels = np.array(points.point_labels)

    masks, scores, logits = masker.get_masks(
        point_coords=point_coords, point_labels=point_labels, multimask_output=multimask_output)

    # Convert masks to list of lists for JSON serialization
    masks_list = [mask.tolist() for mask in masks]

    show_masks(image=image, masks=masks, scores=scores,
               point_coords=point_coords, box_coords=None, input_labels=point_labels, borders=True)

    return {
        "masks": masks_list,
        "scores": scores.tolist(),
        "logits": logits.tolist()
    }
