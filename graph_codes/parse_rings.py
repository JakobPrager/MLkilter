import cv2
import numpy as np
import pandas as pd

# Load both images
image_without_rings = cv2.imread("Blank_align.jpg")
image_with_rings = cv2.imread("test/frame_0000.jpg")

# Resize 'image_with_rings' to match 'image_without_rings'
image_with_rings = cv2.resize(image_with_rings, (image_without_rings.shape[1], image_without_rings.shape[0]))

# Read the center points from CSV
ring_centers = pd.read_csv("graph_codes/ring_centers.csv").values  # Assuming CSV format: x,y

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

# Function to calculate MSE in a specific region
def calculate_mse(region1, region2):
    """Compute Mean Squared Error (MSE) between two regions."""
    diff = np.subtract(region1.astype("float"), region2.astype("float"))
    mse = np.mean(diff ** 2)
    return mse

# Create a copy of the image without rings to draw on
modified_image = image_without_rings.copy()

# Try adding rings and compare MSE in local regions
for (x, y) in ring_centers:
    best_mse = float("inf")
    best_color = None

    # Define region of interest (ROI)
    x1, y1 = max(0, x - ring_radius), max(0, y - ring_radius)
    x2, y2 = min(image_without_rings.shape[1], x + ring_radius), min(image_without_rings.shape[0], y + ring_radius)

    # Extract the region from both images
    original_region = image_with_rings[y1:y2, x1:x2]

    # Try different colors and compute MSE for the local region
    for color_name, color in ring_colors.items():
        temp_image = modified_image.copy()
        cv2.circle(temp_image, (int(x), int(y)), ring_radius, color, ring_thickness)

        # Extract the same region after drawing the ring
        test_region = temp_image[y1:y2, x1:x2]

        # Compute MSE for this region
        mse = calculate_mse(test_region, original_region)

        # Keep track of the best color
        if mse < best_mse:
            best_mse = mse
            best_color = color
    
    # Allow "None" if drawing a ring increases MSE
    original_region_mse = calculate_mse(modified_image[y1:y2, x1:x2], original_region)
    if original_region_mse*1.1 < best_mse :
        best_color = None  # Don't draw the ring

    # Draw only if a valid color was chosen
    if best_color is not None:
        cv2.circle(modified_image, (int(x), int(y)), ring_radius, best_color, ring_thickness)


# Show the final modified image
cv2.imshow("Detected Rings", modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final result
cv2.imwrite("final_detected_rings.jpg", modified_image)
