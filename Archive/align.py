import cv2
import numpy as np

def find_best_match_multi_scale(phone_img, ref_img, scales):
    """Finds the best matching region in the phone image for the reference image across multiple scales."""
    
    # Convert images to grayscale
    gray_phone = cv2.cvtColor(phone_img, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_val = -1  # Initialize best correlation value
    best_scale = None
    for scale in scales:
        # Resize the reference image
        resized_ref = cv2.resize(ref_img, (0, 0), fx=scale, fy=scale)
        gray_resized_ref = cv2.cvtColor(resized_ref, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        try:
            result = cv2.matchTemplate(gray_phone, gray_resized_ref, cv2.TM_CCOEFF_NORMED)
        except:
            print('error',scale, 'too big')
            continue
        # Get the best match position
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print('maxval',max_val)
        # Check if this is the best match so far
        if max_val > best_val:
            best_val = max_val
            best_match = (max_loc, resized_ref.shape[1], resized_ref.shape[0])  # Store location and size
            best_scale = scale
    if best_scale%1 == 0:   
        print('searching for better match')
        #newscales is scale plus minus 40%
        newscales = [scale*0.8, scale*0.9,scale, scale*1.1, scale*1.2]
        best_match = find_best_match_multi_scale(phone_img, ref_img, newscales)[3]
    if best_match:
        top_left, w_ref, h_ref = best_match
        bottom_right = (top_left[0] + w_ref, top_left[1] + h_ref)

        # Draw a rectangle around the matched region
        cv2.rectangle(phone_img, top_left, bottom_right, (0, 255, 0), 2)

    return phone_img, top_left, bottom_right, best_match

# Load images
phone_img = cv2.imread("phone_image.jpg")
ref_img = cv2.imread("cropped_test_image.jpg")

# Define scales for template matching
scales = [1,1.5,2]  # Adjust scales as needed

# Find the best match in the phone image
matched_image, top_left, bottom_right, best_match = find_best_match_multi_scale(phone_img, ref_img, scales)

# Show the result
cv2.imshow("Matched Phone Image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Best match found at: {top_left}, Bottom right corner: {bottom_right}")
