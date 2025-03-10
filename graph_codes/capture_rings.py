import cv2
import numpy as np
import pandas as pd

def calculate_mse(region1, region2):
    """Compute Mean Squared Error (MSE) between two regions."""
    diff = np.subtract(region1.astype("float"), region2.astype("float"))
    mse = np.mean(diff ** 2)
    return mse

def apply_rings(screen_capture, reference_image_path, ring_centers_path):
    """
    Detect rings in the screen capture and apply them to the reference image.
    
    Args:
        screen_capture (numpy.ndarray): The captured screen image (with rings).
        reference_image_path (str): Path to the reference image (without rings).
        ring_centers_path (str): Path to the CSV file containing ring center coordinates.
    
    Returns:
        numpy.ndarray: The modified reference image with applied rings.
    """
    # Load the reference image (without rings)
    reference_image = cv2.imread(reference_image_path)
    
    # Resize reference image to match the screen capture
    screen_capture = cv2.resize(screen_capture, (reference_image.shape[1], reference_image.shape[0]))
    
    # Read the center points from CSV
    ring_centers = pd.read_csv(ring_centers_path).values  # Assuming CSV format: x,y
    
    # Ring parameters
    ring_radius = 20
    ring_thickness = ring_radius // 5
    
    # Define colors in BGR format
    ring_colors = {
        "magenta": (255, 0, 255),
        "blue": (255, 255, 0),
        "green": (0, 255, 0),
        "orange": (0, 165, 255)
    }
    
    # Create a copy of the reference image to draw on
    modified_image = reference_image.copy()
    
    # Apply rings based on MSE comparison
    for (x, y) in ring_centers:
        best_mse = float("inf")
        best_color = None
        
        # Define region of interest (ROI)
        x1, y1 = max(0, x - ring_radius), max(0, y - ring_radius)
        x2, y2 = min(screen_capture.shape[1], x + ring_radius), min(screen_capture.shape[0], y + ring_radius)
        
        # Extract the region from the screen capture
        detected_region = screen_capture[y1:y2, x1:x2]
        
        # Try different colors and compute MSE for the local region
        for color_name, color in ring_colors.items():
            temp_image = np.full_like(detected_region, color, dtype=np.uint8)
            mse = calculate_mse(temp_image, detected_region)
            
            # Keep track of the best color
            #if y < 10 * ring_radius and color_name == "orange":
            #    continue
            if y > 8 * ring_radius and color_name == "magenta":
                continue
            if mse < best_mse:
                best_mse = mse
                best_color = color
        
        # Allow "None" if drawing a ring increases MSE
        original_region_mse = calculate_mse(modified_image[y1:y2, x1:x2], detected_region)
        """if best_color == (0, 165, 255):
            if original_region_mse < best_mse :
                best_color = None  # Don't draw the ring"""
        if original_region_mse*1.1 < best_mse :
            best_color = None  # Don't draw the ring

        # Draw only if a valid color was chosen
        if best_color is not None:
            cv2.circle(modified_image, (int(x), int(y)), ring_radius, best_color, ring_thickness)
        
    return modified_image
