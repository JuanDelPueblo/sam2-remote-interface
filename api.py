from typing import Optional, Union
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
from masker_manager import MaskerManager, MaskerType
import sam2_image_masker as sim
import sam2_video_masker as svm
from utils import *
import base64

app = FastAPI()

masker_manager = MaskerManager()
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

class VideoAddPointsOrBoxRequest(BaseModel):
    frame_idx: int
    obj_id: int
    points: Optional[list[list[float]]] = None
    labels: Optional[list[int]] = None
    clear_old_points: bool = True
    box: Optional[list[float]] = None

class VideoPropagateRequest(BaseModel):
    start_frame_idx: Optional[int] = None
    max_frame_num_to_track: Optional[int] = None
    reverse: bool = False


@app.get("/")
async def root():
    return {"message": "SAM 2 Image Masker API"}


@app.get("/status")
async def status():
    active_masker_type = masker_manager.get_active_masker_type()
    if not active_masker_type:
        return {"status": "inactive", "device": None}

    masker = masker_manager.get_masker()
    if masker:
        return {"status": f"{active_masker_type.value} active", "device": masker.device.type}
    return {"status": "inactive", "device": None}


@app.post("/image/set_image")
async def set_image(request: UploadFile = File(...)):
    masker_manager.set_masker_type(MaskerType.image)
    masker = masker_manager.get_masker()
    assert isinstance(masker, sim.SAM2ImageMasker)
    global image
    image = Image.open(request.file)
    image = np.array(image.convert("RGB"))
    masker.set_image(image)
    return {"message": "Image set successfully"}

@app.post("/image/set_image_batch")
async def set_image_batch(requests: list[UploadFile] = File(...)):
    masker_manager.set_masker_type(MaskerType.image)
    masker = masker_manager.get_masker()
    assert isinstance(masker, sim.SAM2ImageMasker)
    global images
    images = []
    for request in requests:
        img = Image.open(request.file)
        images.append(np.array(img.convert("RGB")))
    masker.set_image_batch(images)
    return {"message": "Image batch set successfully"}

@app.post("/image/get_masks")
async def get_masks(points: ImagePredictorRequest):
    masker = masker_manager.get_masker()
    if not isinstance(masker, sim.SAM2ImageMasker):
        return {"error": "Image masker not active. Call /image/set_image first."}
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

@app.post("/image/get_masks_batch")
async def get_masks_batch(
    data: ImageBatchPredictorRequest
):
    masker = masker_manager.get_masker()
    if not isinstance(masker, sim.SAM2ImageMasker):
        return {"error": "Image masker not active. Call /image/set_image_batch first."}
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

@app.post("/image/reset_predictor")
async def reset_predictor():
    masker = masker_manager.get_masker()
    if not isinstance(masker, sim.SAM2ImageMasker):
        return {"error": "Image masker not active."}
    masker.predictor.reset_predictor()
    return {"message": "Predictor reset successfully"}

@app.post("/video/init_state")
async def init_video_state(video_dir: str):
    masker_manager.set_masker_type(MaskerType.video)
    masker = masker_manager.get_masker()
    assert isinstance(masker, svm.SAM2VideoMasker)
    masker.init_state(video_dir)
    return {"message": "Video state initialized successfully"}

@app.post("/video/reset_state")
async def reset_video_state():
    masker = masker_manager.get_masker()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    masker.reset_state()
    return {"message": "Video state reset successfully"}

@app.post("/video/add_new_points_or_box")
async def add_new_points_or_box(request: VideoAddPointsOrBoxRequest):
    masker = masker_manager.get_masker()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    out_obj_ids, out_mask_logits = masker.add_new_points_or_box(
        frame_idx=request.frame_idx,
        obj_id=request.obj_id,
        points=request.points,
        labels=request.labels,
        clear_old_points=request.clear_old_points,
        box=request.box
    )
    masks_list = [(out_mask_logits[i] > 0.0).cpu().numpy().tolist() for i in range(len(out_obj_ids))]
    return {
        "out_obj_ids": out_obj_ids.tolist(),
        "out_masks": masks_list
    }

@app.post("/video/propagate_in_video")
async def propagate_in_video(request: VideoPropagateRequest):
    masker = masker_manager.get_masker()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    video_segments = masker.propagate_in_video(
        start_frame_idx=request.start_frame_idx,
        max_frame_num_to_track=request.max_frame_num_to_track,
        reverse=request.reverse
    )
    # Convert masks to lists for JSON serialization
    video_segments_serializable = {}
    for frame_idx, obj_dict in video_segments.items():
        video_segments_serializable[frame_idx] = {
            obj_id: mask.tolist() for obj_id, mask in obj_dict.items()
        }
    return {
        "video_segments": video_segments_serializable
    }

@app.post("/video/clear_all_prompts_in_frame")
async def clear_all_prompts_in_frame(frame_idx: int, obj_id: int):
    masker = masker_manager.get_masker()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    masker.clear_all_prompts_in_frame(frame_idx, obj_id)
    return {"message": "Cleared all prompts in frame successfully"}

@app.post("/video/remove_object")
async def remove_object(obj_id: int):
    masker = masker_manager.get_masker()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    masker.remove_object(obj_id)
    return {"message": "Object removed successfully"}