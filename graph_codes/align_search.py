import cv2
import numpy as np

def find_best_scale(phone_img, ref_img, min_scale, max_scale, tolerance=0.01):
    """Finds the best matching scale for the reference image against the phone image."""
    
    best_val = -1
    best_scale = min_scale

    # Convert phone image to grayscale
    gray_phone = cv2.cvtColor(phone_img, cv2.COLOR_BGR2GRAY)

    while max_scale - min_scale > tolerance:
        mid_scale = (min_scale + max_scale) / 2
        # Resize the reference image
        resized_ref = cv2.resize(ref_img, (0, 0), fx=mid_scale, fy=mid_scale)
        gray_resized_ref = cv2.cvtColor(resized_ref, cv2.COLOR_BGR2GRAY)

        # Check if the resized reference image is larger than the phone image
        if gray_resized_ref.shape[0] > gray_phone.shape[0] or gray_resized_ref.shape[1] > gray_phone.shape[1]:
            max_scale = mid_scale  # Decrease the scale if too large
            continue

        # Ensure both images are in the same type
        gray_phone = gray_phone.astype(np.uint8)
        gray_resized_ref = gray_resized_ref.astype(np.uint8)

        # Perform template matching
        result = cv2.matchTemplate(gray_phone, gray_resized_ref, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Update best value and scale
        if max_val > best_val:
            best_val = max_val
            best_scale = mid_scale
            min_scale = mid_scale  # Search for potentially larger scales
        else:
            max_scale = mid_scale  # Search for smaller scales

    return best_scale, best_val

# Load images
phone_img = cv2.imread("phone_image.jpg")
if phone_img is None:
    print("Failed to load phone image.")

ref_img = cv2.imread("cropped_test_image.jpg")
if ref_img is None:
    print("Failed to load reference image.")
else:
    print("Reference image loaded successfully.")

# Proceed only if both images are loaded
if phone_img is not None and ref_img is not None:
    # Set the initial scale range
    min_scale = 0.5  # Start from 10% of the original size
    max_scale = 1.0  # Go up to 200% of the original size

    # Find the best scale for the reference image
    best_scale, best_val = find_best_scale(phone_img, ref_img, min_scale, max_scale)

    print(f"Best scale found: {best_scale}, Best match value: {best_val}")

    # Resize the reference image to the best scale
    optimal_ref = cv2.resize(ref_img, (0, 0), fx=best_scale, fy=best_scale)
    gray_optimal_ref = cv2.cvtColor(optimal_ref, cv2.COLOR_BGR2GRAY)

    # Draw rectangle around the best match
    result = cv2.matchTemplate(phone_img, gray_optimal_ref, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + optimal_ref.shape[1], top_left[1] + optimal_ref.shape[0])
    cv2.rectangle(phone_img, top_left, bottom_right, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Matched Phone Image", phone_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
