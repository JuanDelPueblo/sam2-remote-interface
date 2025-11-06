import base64
import os
import subprocess
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
import json

import requests
from PIL import Image
import numpy as np

# Configuration
BASE_URL = "http://127.0.0.1:8000"
IMAGE_1_PATH = "images/truck.jpg"
IMAGE_2_PATH = "images/groceries.jpg"
VIDEO_DIR = "video"
TRACKING_VIDEO_PATH = "apple.mp4"
OUTPUT_DIR = Path("images")
API_FILE = "api.py"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def wait_for_server(url, timeout=30):
    """Waits for the FastAPI server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            time.sleep(1)
    print("Server failed to start within the timeout period.")
    return False


def set_image(image_path):
    """Calls the /set_image endpoint."""
    url = f"{BASE_URL}/image/set_image"
    response = requests.post(url, json={"path": image_path})
    if response.status_code == 200:
        print(f"Image '{image_path}' set successfully.")
    else:
        print(f"Failed to set image. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200


def set_image_batch(image_paths):
    """Calls the /set_image_batch endpoint."""
    url = f"{BASE_URL}/image/set_image_batch"
    response = requests.post(url, json={"paths": image_paths})
    if response.status_code == 200:
        print(f"Image batch set successfully.")
    else:
        print(f"Failed to set image batch. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200


def get_masks(data, test_name):
    """Calls the /get_masks endpoint and prints the returned mask paths."""
    url = f"{BASE_URL}/image/get_masks"
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_data = response.json()
        saved_paths = response_data.get("saved_mask_paths", [])
        print(f"Test '{test_name}': Successfully retrieved {len(saved_paths)} masks.")
        for path in saved_paths:
            print(f"  Mask saved at: {path}")
    else:
        print(f"Test '{test_name}': Failed to get masks. Status: {response.status_code}, Response: {response.text}")


def reset_predictor():
    """Calls the /reset_predictor endpoint."""
    url = f"{BASE_URL}/image/reset_predictor"
    response = requests.post(url)
    if response.status_code == 200:
        print("Predictor reset successfully.")
    else:
        print(f"Failed to reset predictor. Status: {response.status_code}, Response: {response.text}")


def init_video_state(video_dir):
    """Calls the /video/init_state endpoint."""
    url = f"{BASE_URL}/video/init_state"
    response = requests.post(url, params={"video_frames_dir": video_dir})
    if response.status_code == 200:
        print(f"Video state initialized successfully for '{video_dir}'.")
    else:
        print(f"Failed to initialize video state. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200


def reset_video_state():
    """Calls the /video/reset_state endpoint."""
    url = f"{BASE_URL}/video/reset_state"
    response = requests.post(url)
    if response.status_code == 200:
        print("Video state reset successfully.")
    else:
        print(f"Failed to reset video state. Status: {response.status_code}, Response: {response.text}")


def add_new_points_or_box(frame_idx, obj_id, points=None, labels=None, clear_old_points=True, box=None):
    """Calls the /video/add_new_points_or_box endpoint."""
    url = f"{BASE_URL}/video/add_new_points_or_box"
    payload = {
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "clear_old_points": clear_old_points
    }
    if points is not None:
        payload["points"] = points.tolist() if isinstance(points, np.ndarray) else points
    if labels is not None:
        payload["labels"] = labels.tolist() if isinstance(labels, np.ndarray) else labels
    if box is not None:
        payload["box"] = box.tolist() if isinstance(box, np.ndarray) else box
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Added points/box for frame {frame_idx}, object {obj_id}.")
        print(f"  Object IDs: {response_data.get('out_obj_ids', [])}")
    else:
        print(f"Failed to add points/box. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def propagate_in_video(start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
    """Calls the /video/propagate_in_video endpoint."""
    url = f"{BASE_URL}/video/propagate_in_video"
    payload = {
        "start_frame_idx": start_frame_idx,
        "max_frame_num_to_track": max_frame_num_to_track,
        "reverse": reverse
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        num_frames = len(response_data.get("video_segments", {}))
        saved_paths = response_data.get("saved_mask_paths", {})
        print(f"Propagation successful! Processed {num_frames} frames.")
        print(f"Saved masks for {len(saved_paths)} frames.")
        for frame_idx, paths in saved_paths.items():
            print(f"  Frame {frame_idx}: {paths}")
    else:
        print(f"Failed to propagate. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def get_masks_batch(data, test_name):
    """Calls the /get_masks_batch endpoint and prints the returned mask paths."""
    url = f"{BASE_URL}/image/get_masks_batch"
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_data = response.json()
        saved_paths_batch = response_data.get("saved_mask_paths_batch", [])
        print(f"Test '{test_name}': Successfully retrieved {len(saved_paths_batch)} batches of masks.")
        
        for i, saved_paths in enumerate(saved_paths_batch):
            print(f"  Processing batch {i+1} with {len(saved_paths)} masks.")
            for path in saved_paths:
                print(f"    Mask saved at: {path}")
    else:
        print(f"Test '{test_name}': Failed to get batch masks. Status: {response.status_code}, Response: {response.text}")


def run_tests():
    """Runs the full test suite."""
    # Test cases
    test_cases = [
        {
            "name": "single_point",
            "payload": {"point_coords": [[500, 375]], "point_labels": [1], "multimask_output": False}
        },
        {
            "name": "single_box",
            "payload": {"input_boxes": [[425, 600, 700, 875]], "multimask_output": False}
        },
        {
            "name": "point_and_box",
            "payload": {"point_coords": [[575, 750]], "point_labels": [1], "input_boxes": [[425, 600, 700, 875]], "multimask_output": False}
        }
    ]

    print("=" * 60)
    print("RUNNING IMAGE MASKING TESTS")
    print("=" * 60)

    for test in test_cases:
        print(f"\n--- Running test: {test['name']} ---")
        
        # 1. Set the image for the test
        if not set_image(IMAGE_1_PATH):
            continue # Skip to next test if image setting fails
            
        # 2. Get the mask
        get_masks(test["payload"], test["name"])
        
        # 3. Reset the predictor for the next test
        reset_predictor()
        
        time.sleep(1) # Small delay between tests

    # --- Batch Test ---
    print("\n--- Running test: batch_points ---")
    batch_test_case = {
        "name": "batch_points",
        "payload": {
            "items": [
                {
                    "point_coords": [[[500, 375]], [[650, 750]]],
                    "point_labels": [[1], [1]]
                },
                {
                    "point_coords": [[[400, 300]], [[630, 300]]],
                    "point_labels": [[1], [1]]
                }
            ],
            "multimask_output": False
        }
    }
    # Using the same image twice for the batch
    set_image_batch([IMAGE_1_PATH, IMAGE_2_PATH])
    get_masks_batch(batch_test_case["payload"], batch_test_case["name"])
    reset_predictor()
    time.sleep(1)


def run_video_tests():
    """Runs the video masking test suite."""
    print("\n" + "=" * 60)
    print("RUNNING VIDEO MASKING TESTS")
    print("=" * 60)
    
    # Check if video directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"\nVideo directory '{VIDEO_DIR}' not found. Skipping video tests.")
        return
    
    print(f"\n--- Running test: video_masking ---")
    
    # Initialize video state
    if not init_video_state(VIDEO_DIR):
        print("Failed to initialize video state. Skipping video tests.")
        return
    
    # Test case from user request
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    
    # Let's add a positive click at (x, y) = (210, 350) to get started
    print(f"\nAdding first point at (210, 350) on frame {ann_frame_idx}...")
    points = np.array([[210, 350]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    success, _ = add_new_points_or_box(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )
    
    if not success:
        print("Failed to add first point. Aborting video test.")
        return
    
    time.sleep(0.5)
    
    # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
    # sending all clicks (and their labels) to `add_new_points_or_box`
    print(f"\nAdding second point at (250, 220) on frame {ann_frame_idx}...")
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    success, _ = add_new_points_or_box(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels
    )
    
    if not success:
        print("Failed to add second point. Aborting video test.")
        return
    
    time.sleep(0.5)
    
    # Propagate through the video
    print(f"\nPropagating masks through video...")
    success, result = propagate_in_video()
    
    if success:
        print("\n✓ Video masking test completed successfully!")
    else:
        print("\n✗ Video masking test failed.")
    
    # Reset video state for cleanup
    reset_video_state()


def load_tracking_video(video_path):
    """Calls the /tracking/load_video endpoint."""
    url = f"{BASE_URL}/tracking/load_video"
    response = requests.post(url, json={"video_path": video_path})
    if response.status_code == 200:
        response_data = response.json()
        print(f"Video loaded successfully: {video_path}")
        print(f"  Shape: {response_data.get('shape', [])}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
    else:
        print(f"Failed to load tracking video. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def track_grid(grid_size=15, add_support_grid=True):
    """Calls the /tracking/track_grid endpoint."""
    url = f"{BASE_URL}/tracking/track_grid"
    payload = {
        "grid_size": grid_size,
        "add_support_grid": add_support_grid
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Grid tracking completed successfully!")
        print(f"  Num points: {response_data.get('num_points', 0)}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
        print(f"  Output video: {response_data.get('output_video_path', 'N/A')}")
    else:
        print(f"Failed to track grid. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def track_points(queries, add_support_grid=True):
    """Calls the /tracking/track_points endpoint."""
    url = f"{BASE_URL}/tracking/track_points"
    payload = {
        "queries": queries if isinstance(queries, list) else queries.tolist(),
        "add_support_grid": add_support_grid
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_data = response.json()
        print(f"Point tracking completed successfully!")
        print(f"  Num points: {response_data.get('num_points', 0)}")
        print(f"  Num frames: {response_data.get('num_frames', 0)}")
        print(f"  Output video: {response_data.get('output_video_path', 'N/A')}")
    else:
        print(f"Failed to track points. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200, response.json() if response.status_code == 200 else None


def run_tracking_tests():
    """Runs the tracking test suite."""
    print("\n" + "=" * 60)
    print("RUNNING TRACKING TESTS")
    print("=" * 60)
    
    # Check if tracking video exists
    if not os.path.exists(TRACKING_VIDEO_PATH):
        print(f"\nTracking video '{TRACKING_VIDEO_PATH}' not found. Skipping tracking tests.")
        return
    
    # Test 1: Load video
    print(f"\n--- Test 1: Load tracking video ---")
    success, result = load_tracking_video(TRACKING_VIDEO_PATH)
    
    if not success:
        print("Failed to load tracking video. Skipping remaining tracking tests.")
        return
    
    time.sleep(1)
    
    # Test 2: Track grid
    print(f"\n--- Test 2: Track grid of points ---")
    success, result = track_grid(grid_size=10, add_support_grid=True)
    
    if success:
        print("\n✓ Grid tracking test completed successfully!")
    else:
        print("\n✗ Grid tracking test failed.")
    
    time.sleep(1)
    
    # Test 3: Track specific points
    print(f"\n--- Test 3: Track specific query points ---")
    # Define some query points: [frame, x, y]
    # Let's track a few points starting from frame 0
    queries = [
        [0, 400, 350],  # Center point
        [10, 600, 500],  # Upper-left area
        [20, 750, 600],  # Lower-right area
        [30, 900, 200]
    ]
    success, result = track_points(queries, add_support_grid=False)
    
    if success:
        print("\n✓ Point tracking test completed successfully!")
    else:
        print("\n✗ Point tracking test failed.")
    
    time.sleep(1)
    
    # Test 4: Track points with support grid
    print(f"\n--- Test 4: Track points with support grid ---")
    queries = [
        [0, 320, 240],  # Single point with support grid
    ]
    success, result = track_points(queries, add_support_grid=True)
    
    if success:
        print("\n✓ Point tracking with support grid completed successfully!")
    else:
        print("\n✗ Point tracking with support grid failed.")
    
    print("\n" + "=" * 60)
    print("TRACKING TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Start the FastAPI server as a background process
    server_process = subprocess.Popen(["fastapi", "dev", API_FILE])
    print(f"Starting FastAPI server with PID: {server_process.pid}...")

    try:
        # Wait for the server to be ready
        # The root endpoint in api.py returns a simple message
        if wait_for_server(f"{BASE_URL}/", timeout=30):
            # Run the image tests
            run_tests()
            
            # Run the video tests
            run_video_tests()
            
            # Run the tracking tests
            run_tracking_tests()
            
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED")
            print("=" * 60)
        else:
            print("Could not connect to the server. Aborting tests.")

    finally:
        # Stop the server
        print("\nShutting down the server...")
        server_process.terminate()
        try:
            # Wait for the process to terminate
            server_process.wait(timeout=10)
            print("Server shut down successfully.")
        except subprocess.TimeoutExpired:
            print("Server did not terminate in time, killing it.")
            server_process.kill()
            print("Server killed.")
