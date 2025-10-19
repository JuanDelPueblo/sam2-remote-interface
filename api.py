from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
import sam2_image_masker as sim
from utils import *
import os
import time
import secrets
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static serving for generated mask images
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

masker = sim.SAM2ImageMasker()
image = None  # Global variable to store the current image


class ImagePredictorRequest(BaseModel):
    point_coords: Optional[list[list[float]]] = None
    point_labels: Optional[list[int]] = None
    input_boxes: Optional[list[list[float]]] = None
    multimask_output: bool = True


@app.get("/")
async def root():
    return {"message": "SAM 2 Image Masker API"}

@app.get("/status")
async def status():
    if masker.sam2_model is None:
        return {"status": "not ok", "device": None}
    
    return {"status": "ok", "device": masker.device.type}

@app.post("/set_image")
async def set_image(request: UploadFile = File(...)):
    global image
    image = Image.open(request.file)
    image = np.array(image.convert("RGB"))
    masker.set_image(image)
    return {"message": "Image set successfully"}


@app.post("/get_masks")
async def get_masks(points: ImagePredictorRequest, request: Request):
    global image
    if image is None:
        return {"error": "No image set. Call /set_image first."}
    point_coords = None
    point_labels = None
    input_boxes = None
    multimask_output = points.multimask_output

    if points.point_coords is not None:
        point_coords = np.array(points.point_coords)
    if points.point_labels is not None:
        point_labels = np.array(points.point_labels)
    if points.input_boxes is not None:
        input_boxes = np.array(points.input_boxes)

    masks, scores, logits = masker.get_masks(
        point_coords=point_coords, point_labels=point_labels, input_boxes=input_boxes, multimask_output=multimask_output)

    # Convert masks to list of lists for JSON serialization
    masks_list = [mask.tolist() for mask in masks]

    # Save overlay images and return them as static URLs
    # Create a unique prefix to avoid clashes across requests
    prefix = f"mask_{int(time.time())}_{secrets.token_hex(4)}"
    saved_paths = show_masks(image=image, masks=masks, scores=scores,
                             point_coords=point_coords, box_coords=input_boxes, input_labels=point_labels, borders=True,
                             out_dir=os.path.join(os.path.dirname(__file__), "images"), prefix=prefix)

    # Build public URLs for each saved image
    mask_image_urls = []
    for p in saved_paths:
        filename = os.path.basename(p)
        url = str(request.url_for("images", path=filename))
        mask_image_urls.append(url)

    return {
        "masks": masks_list,
        "scores": scores.tolist(),
        "logits": logits.tolist(),
        "mask_image_urls": mask_image_urls
    }

@app.post("/reset_predictor")
async def reset_predictor():
    masker.predictor.reset_predictor()
    return {"message": "Predictor reset successfully"}