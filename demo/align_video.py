import cv2
import numpy as np

def overlay_image(frame, ref_img, top_left, bottom_right, alpha=0.1):
    """Overlays the reference image onto the frame at the detected position."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Resize reference image to match the detected area
    overlay_resized = cv2.resize(ref_img, (x2 - x1, y2 - y1))

    # Extract region of interest (ROI) from frame
    roi = frame[y1:y2, x1:x2]
    # Blend images using alpha transparency
    blended = cv2.addWeighted(roi, 1 - alpha, overlay_resized, alpha, 0)

    # Replace the ROI in the frame with the blended image
    frame[y1:y2, x1:x2] = blended
    return frame

def find_best_match_multi_scale(phone_img, ref_img, scales, roi=None):
    """Finds the best matching region in the phone image for the reference image across multiple scales.
       If roi is provided, it restricts the search area to improve speed."""
    
    gray_phone = cv2.cvtColor(phone_img, cv2.COLOR_BGR2GRAY)
    if roi is not None:
        x, y, w, h = roi
        gray_phone = gray_phone[y:y+h, x:x+w]  # Crop to ROI

    best_match = None
    best_val = -1  
    best_scale = None

    for scale in scales:
        try:
            resized_ref = cv2.resize(ref_img, (0, 0), fx=scale, fy=scale)
            gray_resized_ref = cv2.cvtColor(resized_ref, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(gray_phone, gray_resized_ref, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_match = (max_loc, resized_ref.shape[1], resized_ref.shape[0])  
                best_scale = scale
        except:
            continue  

    if best_match:
        (x_offset, y_offset), w_ref, h_ref = best_match
        if roi is not None:
            x_offset += roi[0]  # Adjust if using ROI
            y_offset += roi[1]

        top_left = (x_offset, y_offset)
        bottom_right = (top_left[0] + w_ref, top_left[1] + h_ref)
        return top_left, bottom_right, best_scale, best_val
    return None, None, None, -1  # No match found

# Open video capture
cap = cv2.VideoCapture("kiltervideo.mp4")  # Change to 0 for webcam

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    cap.release()
    exit()

print("Frame shape:", frame.shape)
# Load reference image
ref_img = cv2.imread("cropped_test_image.jpg")

# Define scales
scales = [0.7,0.8,0.9]

top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame, ref_img, scales)

if top_left is None:
    print("No initial match found.")
    cap.release()
    exit()

# Parameters
CONFIDENCE_THRESHOLD = 0.1  # If below this, do a full re-search
FULL_SEARCH_INTERVAL = 1000   # Every N frames, do a global search
frame_count = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    frame_count += 1

    # Define a region of interest (ROI) around the last found position
    roi_margin = 50  
    x, y = top_left
    w, h = bottom_right[0] - x, bottom_right[1] - y
    roi = (max(0, x - roi_margin), max(0, y - roi_margin), min(frame.shape[1], w + 2*roi_margin), min(frame.shape[0], h + 2*roi_margin))
    # Search within ROI first
    top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame, ref_img, [best_scale * 0.9, best_scale, best_scale * 1.1], roi)
    
    # If confidence is too low OR it's time for a full search, do a full re-search
    if best_val < CONFIDENCE_THRESHOLD or frame_count % FULL_SEARCH_INTERVAL == 0:
        print("Resetting with full search...",roi)
        top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame, ref_img, scales)  # Full search
        frame_count = 0  # Reset counter

    if top_left:
        print("Found at:", top_left, bottom_right,frame.shape)
        #frame = overlay_image(frame, ref_img, top_left, bottom_right, alpha=0.3)  # Adjust transparency
        frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw rectangle
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
