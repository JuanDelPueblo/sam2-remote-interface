
import base64
import os
import subprocess
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# Configuration
BASE_URL = "http://127.0.0.1:8000"
IMAGE_PATH = "truck.jpg"
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
    url = f"{BASE_URL}/set_image"
    with open(image_path, "rb") as f:
        files = {"request": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files)
    if response.status_code == 200:
        print(f"Image '{image_path}' set successfully.")
    else:
        print(f"Failed to set image. Status: {response.status_code}, Response: {response.text}")
    return response.status_code == 200


def get_masks(data, test_name):
    """Calls the /get_masks endpoint and saves the returned masks."""
    url = f"{BASE_URL}/get_masks"
    response = requests.post(url, json=data)

    if response.status_code == 200:
        masks_data = response.json().get("mask_images_base64", [])
        print(f"Test '{test_name}': Successfully retrieved {len(masks_data)} masks.")
        for i, mask_b64 in enumerate(masks_data):
            try:
                # Decode the base64 string
                mask_bytes = base64.b64decode(mask_b64)
                
                # Create an image from the bytes
                mask_image = Image.open(BytesIO(mask_bytes))
                
                # Generate a unique filename
                timestamp = int(datetime.now().timestamp())
                filename = f"mask_{timestamp}_{test_name}_{i+1}.png"
                output_path = OUTPUT_DIR / filename
                
                # Save the image
                mask_image.save(output_path)
                print(f"Saved mask to '{output_path}'")

            except Exception as e:
                print(f"Error processing mask for test '{test_name}': {e}")
    else:
        print(f"Test '{test_name}': Failed to get masks. Status: {response.status_code}, Response: {response.text}")


def reset_predictor():
    """Calls the /reset_predictor endpoint."""
    url = f"{BASE_URL}/reset_predictor"
    response = requests.post(url)
    if response.status_code == 200:
        print("Predictor reset successfully.")
    else:
        print(f"Failed to reset predictor. Status: {response.status_code}, Response: {response.text}")


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
            "payload": {"point_coords": [[500, 375]], "point_labels": [1], "input_boxes": [[425, 600, 700, 875]], "multimask_output": False}
        }
    ]

    for test in test_cases:
        print(f"\n--- Running test: {test['name']} ---")
        
        # 1. Set the image for the test
        if not set_image(IMAGE_PATH):
            continue # Skip to next test if image setting fails
            
        # 2. Get the mask
        get_masks(test["payload"], test["name"])
        
        # 3. Reset the predictor for the next test
        reset_predictor()
        
        time.sleep(1) # Small delay between tests


if __name__ == "__main__":
    # Start the FastAPI server as a background process
    server_process = subprocess.Popen(["fastapi", "dev", API_FILE])
    print(f"Starting FastAPI server with PID: {server_process.pid}...")

    try:
        # Wait for the server to be ready
        # The root endpoint in api.py returns a simple message
        if wait_for_server(f"{BASE_URL}/", timeout=30):
            # Run the tests
            run_tests()
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
