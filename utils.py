import numpy as np
import cv2


def show_mask(image, mask, random_color=False, borders=True):
    if random_color:
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
    else:
        color = np.array([255, 144, 30], dtype=np.uint8)  # BGR for blue
    
    h, w = mask.shape[-2:]
    mask_bool = mask.astype(bool)

    # Create a colored mask
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask_bool] = color

    # Blend the colored mask with the original image
    # The alpha channel is simulated by weighting
    image[mask_bool] = cv2.addWeighted(image[mask_bool], 0.5, color_mask[mask_bool], 0.5, 0)

    if borders:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=2)
    return image


def show_points(image, coords, labels, marker_size=20):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for p in pos_points:
        cv2.drawMarker(image, (int(p[0]), int(p[1])), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    for p in neg_points:
        cv2.drawMarker(image, (int(p[0]), int(p[1])), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    return image


def show_box(image, boxes):
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return image


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Render mask overlay images using OpenCV and return them as byte arrays.

    Parameters:
    - image: numpy array (H,W,3) RGB image to display under masks
    - masks: iterable of binary numpy arrays (H,W) for each mask
    - scores: iterable of floats corresponding to each mask
    - point_coords, box_coords, input_labels: optional annotations to render
    - borders: whether to draw contours around masks

    Returns: list of byte arrays, each containing a PNG-encoded image
    """
    output_images = []
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Create a fresh copy for each mask's image
        output_image = image_bgr.copy()

        # Apply mask
        output_image = show_mask(
            output_image, mask, random_color=True, borders=borders)

        # Draw points and boxes if they exist
        if point_coords is not None:
            assert input_labels is not None
            output_image = show_points(
                output_image, point_coords, input_labels)
        if box_coords is not None:
            output_image = show_box(output_image, box_coords)

        # Add score text
        if len(scores) > 1:
            text = f"Mask {i+1}, Score: {score:.3f}"
            cv2.putText(output_image, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the image to a byte array
        _, buffer = cv2.imencode('.png', output_image)
        output_images.append(buffer.tobytes())

    return output_images
