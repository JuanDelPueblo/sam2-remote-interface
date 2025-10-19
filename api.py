from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
import sam2_image_masker as sim
from utils import *
import base64

app = FastAPI()

masker = sim.SAM2ImageMasker()
image = None


class ImagePredictorRequest(BaseModel):
    point_coords: Optional[list[list[float]]] = None
    point_labels: Optional[list[int]] = None
    input_boxes: Optional[list[list[float]]] = None
    multimask_output: bool = False


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

    masks_list = [mask.tolist() for mask in masks]

    mask_images = show_masks(image=image, masks=masks, scores=scores,
                             point_coords=point_coords, box_coords=input_boxes, input_labels=point_labels, borders=True)

    mask_images_base64 = [base64.b64encode(
        img).decode('utf-8') for img in mask_images]

    return {
        "masks": masks_list,
        "scores": scores.tolist(),
        "logits": logits.tolist(),
        "mask_images_base64": mask_images_base64
    }


@app.post("/reset_predictor")
async def reset_predictor():
    masker.predictor.reset_predictor()
    return {"message": "Predictor reset successfully"}