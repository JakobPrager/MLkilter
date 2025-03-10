import cv2
import numpy as np

# Load the grid image (the one you've generated)
grid_image = cv2.imread("test/rez_0000.jpg")  # Change this to your grid image path
# Load the input image (the one you want to align)
input_image = cv2.imread('blank.PNG')  # Change this to your input image path

# Initialize parameters for the overlay (default values)
scale_factor = 1.0  # Initial size scale
offset_x = 0        # Initial horizontal position
offset_y = 0        # Initial vertical position

def update_overlay(val):
    """Callback function to update the overlay based on slider values."""
    global scale_factor, offset_x, offset_y

    # Get updated values from the sliders
    scale_factor = cv2.getTrackbarPos('Scale', 'Alignment') / 1000  # Scale factor
    offset_x = cv2.getTrackbarPos('Offset X', 'Alignment')  # Horizontal position
    offset_y = cv2.getTrackbarPos('Offset Y', 'Alignment')  # Vertical position

    # Create a copy of the grid image to overlay the input image
    overlay_image = grid_image.copy()

    # Resize the input image based on the scale factor
    height, width = input_image.shape[:2]
    resized_input = cv2.resize(input_image, (int(width * scale_factor), int(height * scale_factor)))

    # Check if the resized image fits in the grid (at given offset positions)
    if (offset_y + resized_input.shape[0] > grid_image.shape[0]) or (offset_x + resized_input.shape[1] > grid_image.shape[1]):
        print("Resized image does not fit in the grid at the given position.")
        return

    # Position the resized image on top of the grid image
    h_offset, w_offset = offset_y, offset_x
    opacity = 0.5  # Opacity of the overlay
    for c in range(0, 3):
        overlay_image[h_offset:h_offset + resized_input.shape[0], w_offset:w_offset + resized_input.shape[1], c] = \
            (1 - opacity) * overlay_image[h_offset:h_offset + resized_input.shape[0], w_offset:w_offset + resized_input.shape[1], c] + \
            opacity * resized_input[:, :, c]
    # Show the updated overlay
    cv2.imshow('Alignment', overlay_image)

def setup_sliders():
    """Setup sliders for adjusting overlay size and position."""
    cv2.namedWindow('Alignment')
    
    # Create sliders for manual adjustment
    cv2.createTrackbar('Scale', 'Alignment', 1000, 2000, update_overlay)  # Scale from 0.5x to 2x size
    cv2.createTrackbar('Offset X', 'Alignment', 0, grid_image.shape[1], update_overlay)  # Horizontal offset
    cv2.createTrackbar('Offset Y', 'Alignment', 0, grid_image.shape[0], update_overlay)  # Vertical offset

    # Initial update of overlay
    update_overlay(0)

# Run the setup for sliders and overlay
setup_sliders()

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
