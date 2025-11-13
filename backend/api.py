from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model_manager import ModelManager, ModelType
import sam2_video_masker as svm
import co_tracker as cot
from utils import *
import os
from pathlib import Path
from datetime import datetime
import mediapy

app = FastAPI()

model_manager = ModelManager()
video_dir: Optional[str] = None
tracking_video: Optional[np.ndarray] = None
tracking_video_path: Optional[str] = None


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


class TrackingLoadVideoRequest(BaseModel):
    video_path: str


class TrackingGridRequest(BaseModel):
    grid_size: int = 15
    add_support_grid: bool = True


class TrackingPointsRequest(BaseModel):
    queries: list[list[float]]  # List of [t, x, y] coordinates
    add_support_grid: bool = True


@app.get("/")
async def root():
    return {"message": "SAM 2 Image Masker API"}


@app.get("/status")
async def status():
    active_model_type = model_manager.get_active_model_type()
    if not active_model_type:
        return {"status": "inactive", "device": None}

    model = model_manager.get_model()
    if model:
        return {"status": f"{active_model_type.value} active", "device": model.device.type}
    return {"status": "inactive", "device": None}


@app.post("/video/init_state")
async def init_video_state(video_frames_dir: str):
    model_manager.set_model_type(ModelType.video)
    masker = model_manager.get_model()
    assert isinstance(masker, svm.SAM2VideoMasker)
    global video_dir
    video_dir = video_frames_dir
    masker.init_state(video_dir)
    return {"message": "Video state initialized successfully"}

@app.post("/video/reset_state")
async def reset_video_state():
    masker = model_manager.get_model()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    masker.reset_state()
    return {"message": "Video state reset successfully"}

@app.post("/video/add_new_points_or_box")
async def add_new_points_or_box(request: VideoAddPointsOrBoxRequest):
    masker = model_manager.get_model()
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
        "out_obj_ids": out_obj_ids,
        "out_masks": masks_list
    }

@app.post("/video/propagate_in_video")
async def propagate_in_video(request: VideoPropagateRequest):
    masker = model_manager.get_model()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    global video_dir
    if video_dir is None:
        return {"error": "Video directory not set. Call /video/init_state first."}
    
    video_segments = masker.propagate_in_video(
        start_frame_idx=request.start_frame_idx,
        max_frame_num_to_track=request.max_frame_num_to_track,
        reverse=request.reverse
    )
    
    # Save masks for each frame
    saved_mask_paths = save_video_masks(video_dir, video_segments)
    
    # Convert masks to lists for JSON serialization
    video_segments_serializable = {}
    for frame_idx, obj_dict in video_segments.items():
        video_segments_serializable[frame_idx] = {
            obj_id: mask.tolist() for obj_id, mask in obj_dict.items()
        }
    return {
        "video_segments": video_segments_serializable,
        "saved_mask_paths": saved_mask_paths
    }

@app.post("/video/clear_all_prompts_in_frame")
async def clear_all_prompts_in_frame(frame_idx: int, obj_id: int):
    masker = model_manager.get_model()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    masker.clear_all_prompts_in_frame(frame_idx, obj_id)
    return {"message": "Cleared all prompts in frame successfully"}

@app.post("/video/remove_object")
async def remove_object(obj_id: int):
    masker = model_manager.get_model()
    if not isinstance(masker, svm.SAM2VideoMasker):
        return {"error": "Video masker not active."}
    masker.remove_object(obj_id)
    return {"message": "Object removed successfully"}


@app.post("/tracking/load_video")
async def load_tracking_video(request: TrackingLoadVideoRequest):
    """Load a video file for tracking."""
    model_manager.set_model_type(ModelType.tracking)
    tracker = model_manager.get_model()
    assert isinstance(tracker, cot.CoTracker)
    
    global tracking_video, tracking_video_path
    tracking_video_path = request.video_path
    
    # Load video using mediapy
    tracking_video = mediapy.read_video(request.video_path)
    
    return {
        "message": "Video loaded successfully",
        "shape": tracking_video.shape,
        "num_frames": tracking_video.shape[0]
    }


@app.post("/tracking/track_grid")
async def track_grid(request: TrackingGridRequest):
    """Track a grid of points across the video."""
    tracker = model_manager.get_model()
    if not isinstance(tracker, cot.CoTracker):
        return {"error": "Tracker not active. Call /tracking/load_video first."}
    
    global tracking_video, tracking_video_path
    if tracking_video is None:
        return {"error": "No video loaded. Call /tracking/load_video first."}
    
    # Run tracking
    tracks, visibility = tracker.track(
        tracking_video, 
        queries=None, 
        grid_size=request.grid_size,
        add_support_grid=request.add_support_grid
    )
    
    # Save visualization
    painted_video = cot.paint_point_track(tracking_video, tracks, visibility)
    
    # Save output video
    video_name = Path(tracking_video_path).stem
    output_dir = Path(os.path.dirname(tracking_video_path))
    timestamp = int(datetime.now().timestamp())
    output_filename = f"{video_name}_tracked_grid_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    fps = 30  # Default fps
    try:
        original_fps = mediapy.read_video(tracking_video_path).metadata.fps
        if original_fps:
            fps = original_fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)
    
    return {
        "message": "Grid tracking completed",
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "num_points": tracks.shape[0],
        "num_frames": tracks.shape[1],
        "output_video_path": str(output_path)
    }


@app.post("/tracking/track_points")
async def track_points(request: TrackingPointsRequest):
    """Track specific query points across the video."""
    tracker = model_manager.get_model()
    if not isinstance(tracker, cot.CoTracker):
        return {"error": "Tracker not active. Call /tracking/load_video first."}
    
    global tracking_video, tracking_video_path
    if tracking_video is None:
        return {"error": "No video loaded. Call /tracking/load_video first."}
    
    # Convert queries to numpy array
    queries = np.array(request.queries)
    
    # Run tracking
    tracks, visibility = tracker.track(
        tracking_video,
        queries=queries,
        add_support_grid=request.add_support_grid
    )
    
    # Save visualization
    painted_video = cot.paint_point_track(tracking_video, tracks, visibility)
    
    # Save output video
    video_name = Path(tracking_video_path).stem
    output_dir = Path(os.path.dirname(tracking_video_path))
    timestamp = int(datetime.now().timestamp())
    output_filename = f"{video_name}_tracked_points_{timestamp}.mp4"
    output_path = output_dir / output_filename
    
    fps = 30  # Default fps
    try:
        original_fps = mediapy.read_video(tracking_video_path).metadata.fps
        if original_fps:
            fps = original_fps
    except:
        pass
    
    mediapy.write_video(str(output_path), painted_video, fps=fps)
    
    return {
        "message": "Point tracking completed",
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "num_points": tracks.shape[0],
        "num_frames": tracks.shape[1],
        "output_video_path": str(output_path)
    }
