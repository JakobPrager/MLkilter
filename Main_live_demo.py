import cv2
import numpy as np
import torch
from MLbackend.Climb_CNN import ClimbingGradeCNN
import torch
import torchvision.transforms as transforms
from PIL import Image
from graph_codes.capture_rings import apply_rings


model = ClimbingGradeCNN()  # Initialize your model architecture
model.load_state_dict(torch.load('src/climbing_grade_cnn_aug.pth',weights_only=True))  # Load state dict from the .pth file
model.eval()

def preprocess_image(image, frame_count):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the numpy array (or tensor) to a PIL Image
    im = transforms.ToPILImage()(image_rgb)

    #if frame_count % 100 == 0:
        # Save the image directly using the save method
        #im.save(f"recordtest{frame_count}.jpg")
        
    
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])
    return transform(image_rgb).unsqueeze(0)

def classify_boulder_grade(image,frame_count):
    """Classifies the boulder grade using the CNN model."""
    ring_image = apply_rings(image, "src/Blank_align.jpg", "graph_codes/ring_centers.csv")
    processed_image = preprocess_image(ring_image, frame_count)
    

    # Predict the grade
    predictions = model(processed_image)
    """image = Image.open(f'rectangle40/frame_{frame_count:04d}.jpg').convert('RGB')
    #image = transforms.ToPILImage()(image)
    transform = transforms.Compose([
        #transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])
    imga = transform(image)
    predictions = model(imga.unsqueeze(0))"""
    predicted_grade = torch.round(predictions)
    grade_order = ['4a', '4b','4c','5a', '5b', '5c', '6a', '6b', '6c', '7a','7b','7c','8a','8b', '8c']
    predicted_grade = grade_order[int(predicted_grade[0][0])]
    return predicted_grade, ring_image

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

# Open Mac webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set higher resolution if needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to access webcam.")
    cap.release()
    exit()

# Load reference image
#ref_img = cv2.imread("cropped_test_image.jpg")
#frame_height, frame_width = frame.shape[:2]
ref_img = cv2.imread("src/Blank_align.jpg")
# Resize reference to match webcam resolution
#ref_img = cv2.resize(ref_img, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

# Define scales
scales = [0.4,0.5,0.6]

# Full search for first frame
top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame, ref_img, scales)
predicted_grade = 'None'
ring_image = cv2.imread("src/Blank_align.jpg") 
if top_left is None:
    print("No initial match found.")
    cap.release()
    exit()

# Parameters
CONFIDENCE_THRESHOLD = 0.15  # If below this, do a full re-search
FULL_SEARCH_INTERVAL = 50   # Every N frames, do a global search
frame_count = 0  # Initialize the frame counter

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
    if best_val < CONFIDENCE_THRESHOLD:
        print("Resetting with full search...")
        top_left, bottom_right, best_scale, best_val = find_best_match_multi_scale(frame, ref_img, scales)  # Full search
        frame_count = 0  # Reset counter
        cv2.putText(frame, f'Looking for route', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frame = overlay_image(frame, ref_img, top_left, bottom_right, alpha=0.2)  # Adjust transparency
        frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw rectangle

    elif top_left:
        # Extract the ROI for classification
        boundary_margin = 0  # Increase or decrease this value as needed
        x1 = max(0, top_left[0] - boundary_margin)
        y1 = max(0, top_left[1])
        x2 = min(frame.shape[1], bottom_right[0] + boundary_margin)
        y2 = min(frame.shape[0], bottom_right[1])
        roi_image = frame[y1:y2, x1:x2]

        # Classify the boulder grade every 10th frame (not skipping frames)
        predicted_grade, ring_image = classify_boulder_grade(roi_image, frame_count)
        
        cv2.putText(frame, f'Grade: {predicted_grade}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Resize the ROI image to a smaller size for display
        roi_display = cv2.resize(ring_image, (160, 160))  # Resize to 160x160 pixels

        # Overlay the ROI image on the main frame at the bottom right corner
        h, w = frame.shape[:2]
        frame[h-160:h, w-160:w] = roi_display  # Place it in the bottom right corner

        frame = overlay_image(frame, ref_img, top_left, bottom_right, alpha=0.2)  # Adjust transparency
        frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw rectangle

    # Show the current frame continuously, without skipping
    cv2.imshow("Live Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break  

cap.release()
cv2.destroyAllWindows()
