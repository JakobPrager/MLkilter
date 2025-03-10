import cv2
import numpy as np
import torch
from MLbackend.Climb_CNN import ClimbingGradeCNN
import torchvision.transforms as transforms
import mss

# Initialize your model architecture
model = ClimbingGradeCNN()
model.load_state_dict(torch.load('src/climbing_grade_cnn.pth', weights_only=True))
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])
    return transform(image).unsqueeze(0)

def classify_boulder_grade(image):
    """Classifies the boulder grade using the CNN model."""
    #processed_image = preprocess_image(image)
    #predictions = model(processed_image[])
    #predicted_grade = predictions.argmax(dim=1)
    #grade_order = ['4a', '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '7a', '7b', '7c', '8a', '8b', '8c']
    #predicted_grade = grade_order[predicted_grade.item()]
    return 'hi'

def overlay_image(frame, ref_img, top_left, bottom_right, alpha=0.6):
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
    """Finds the best matching region in the phone image for the reference image across multiple scales."""
    
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

# Use mss to capture the screen
with mss.mss() as sct:
    # Define the screen area to capture (you can adjust this as needed)
    monitor = sct.monitors[1]  # Capture the primary screen (1 for the first monitor)
    
    # Load reference image
    ref_img = cv2.imread("cropped_test_image.jpg")

    # Define scales
    scales = [0.4, 0.5, 0.6]

    # Full search for the first frame
    frame = np.array(sct.grab(monitor))  # Capture the first frame
    top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame[:, :, :3], ref_img, scales)

    if top_left is None:
        print("No initial match found.")
        exit()

    # Parameters
    CONFIDENCE_THRESHOLD = 0.15  # If below this, do a full re-search
    FULL_SEARCH_INTERVAL = 50   # Every N frames, do a global search
    frame_count = 0  

    while True:
        # Capture the screen
        frame = np.array(sct.grab(monitor))

        frame_count += 1

        # Define a region of interest (ROI) around the last found position
        roi_margin = 50  
        x, y = top_left
        w, h = bottom_right[0] - x, bottom_right[1] - y
        roi = (max(0, x - roi_margin), max(0, y - roi_margin), min(frame.shape[1], w + 2*roi_margin), min(frame.shape[0], h + 2*roi_margin))

        # Search within ROI first
        top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame[:, :, :3], ref_img, [best_scale * 0.9, best_scale, best_scale * 1.1], roi)

        # If confidence is too low OR it's time for a full search, do a full re-search
        if best_val < CONFIDENCE_THRESHOLD:
            print("Resetting with full search...")
            top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame[:, :, :3], ref_img, scales)  # Full search
            frame_count = 0  # Reset counter

        if top_left:
            # Extract the ROI for classification
            x1, y1 = top_left
            x2, y2 = bottom_right
            roi_image = frame[y1:y2, x1:x2]

            # Classify the boulder grade
            predicted_grade = classify_boulder_grade(roi_image)
            cv2.putText(frame, f'Grade: {predicted_grade}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Resize the ROI image to a smaller size for display
            roi_display = cv2.resize(roi_image, (160, 160))  # Resize to 160x160 pixels

            # Overlay the ROI image on the main frame at the bottom right corner
            h, w = frame.shape[:2]
            frame[h-160:h, w-160:w] = roi_display  # Place it in the bottom right corner

            frame = overlay_image(frame, ref_img, top_left, bottom_right, alpha=0.1)  # Adjust transparency
            frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw rectangle

        cv2.imshow("Live Tracking", cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))  # Convert to BGR for display
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break  

cv2.destroyAllWindows()
