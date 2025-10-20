from typing import Optional
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import sam2_image_masker as sim
from utils import *
import base64

app = FastAPI()

masker = sim.SAM2ImageMasker()
image = None
images = None


class ImagePredictorRequest(BaseModel):
    point_coords: Optional[list[list[float]]] = None
    point_labels: Optional[list[int]] = None
    input_boxes: Optional[list[list[float]]] = None
    multimask_output: bool = False
    return_masked_images: bool = False

class BatchItem(BaseModel):
    point_coords: Optional[list[list[list[float]]]] = None
    point_labels: Optional[list[list[int]]] = None
    input_boxes: Optional[list[list[list[float]]]] = None


class ImageBatchPredictorRequest(BaseModel):
    items: list[BatchItem]
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

@app.post("/set_image_batch")
async def set_image_batch(requests: list[UploadFile] = File(...)):
    global images
    images = []
    for request in requests:
        img = Image.open(request.file)
        images.append(np.array(img.convert("RGB")))
    masker.set_image_batch(images)
    return {"message": "Image batch set successfully"}

@app.post("/get_masks")
async def get_masks(points: ImagePredictorRequest):
    global image
    if image is None:
        return {"error": "No image set. Call /set_image first."}
    point_coords = None
    point_labels = None
    input_boxes = None
    multimask_output = points.multimask_output
    return_masked_images = points.return_masked_images

    if points.point_coords is not None:
        point_coords = np.array(points.point_coords)
    if points.point_labels is not None:
        point_labels = np.array(points.point_labels)
    if points.input_boxes is not None:
        input_boxes = np.array(points.input_boxes)

    masks, scores, logits = masker.predict(
        point_coords=point_coords, point_labels=point_labels, input_boxes=input_boxes, multimask_output=multimask_output)

    masks_list = [mask.tolist() for mask in masks]

    if not return_masked_images:
        return {
            "masks": masks_list,
            "scores": scores.tolist(),
            "logits": logits.tolist()
        }

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

@app.post("/get_masks_batch")
async def get_masks_batch(
    data: ImageBatchPredictorRequest
):
    global images
    if images is None:
        return {"error": "No images set. Call /set_image_batch first."}
    request_data = data
    point_coords_batch = [np.array(item.point_coords) if item.point_coords else None for item in request_data.items]
    point_labels_batch = [np.array(item.point_labels) if item.point_labels else None for item in request_data.items]
    input_boxes_batch = [np.array(item.input_boxes) if item.input_boxes else None for item in request_data.items]

    masks_batch, scores_batch, logits_batch = masker.predict_batch(
        point_coords_batch=point_coords_batch,
        point_labels_batch=point_labels_batch,
        input_boxes_batch=input_boxes_batch,
        multimask_output=request_data.multimask_output
    )

    # Convert results to lists for JSON serialization
    masks_list = [[mask.tolist() for mask in masks] for masks in masks_batch]
    scores_list = [[score.tolist() for score in scores] for scores in scores_batch]
    logits_list = [[logit.tolist() for logit in logits] for logits in logits_batch]

    mask_images_base64_batch = []
    for i, masks in enumerate(masks_batch):
        mask_images = show_masks(
            image=images[i],
            masks=masks,
            scores=scores_batch[i],
            point_coords=point_coords_batch[i],
            box_coords=input_boxes_batch[i],
            input_labels=point_labels_batch[i],
            borders=True
        )
        mask_images_base64 = [base64.b64encode(img).decode('utf-8') for img in mask_images]
        mask_images_base64_batch.append(mask_images_base64)

    return {
        "masks_batch": masks_list,
        "scores_batch": scores_list,
        "logits_batch": logits_list,
        "mask_images_base64_batch": mask_images_base64_batch
    }

@app.post("/reset_predictor")
async def reset_predictor():
    masker.predictor.reset_predictor()
    return {"message": "Predictor reset successfully"}