import cv2
import numpy as np

# Read the image
frame_inv = cv2.imread('test_image.jpg')

# Convert BGR to HSV
hsv = cv2.cvtColor(frame_inv, cv2.COLOR_BGR2HSV)
blur = cv2.GaussianBlur(hsv, (5, 5), 0)

# Define range of color in HSV
lower_red = np.array([90 - 10, 70, 50])
upper_red = np.array([90 + 10, 255, 255])

# Threshold the HSV image to get only the specified color
mask = cv2.inRange(blur, lower_red, upper_red)

# Apply morphological operations
kernel = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(mask, kernel)

# Find contours
contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a black mask
mask_circles = np.zeros_like(frame_inv, dtype=np.uint8)

def scan_subregions(sub_mask, x, y, w, h):
    sub_contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for sc in sub_contours:
        (sx, sy), sr = cv2.minEnclosingCircle(sc)
        center = (int(sx), int(sy))
        sr = int(sr)
        cv2.circle(frame_inv, center, sr, (0, 255, 0), 2)
        cv2.circle(mask_circles, center, sr, (255, 255, 255), -1)

for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    (cx, cy), r = cv2.minEnclosingCircle(c)
    center = (int(cx), int(cy))
    r = int(r)
    
    if r > 1000:  # Large circle, check for subregions
        if w > h:  # Horizontal shape, split vertically
            left_half = mask[y:y+h, x:x+w//2]
            right_half = mask[y:y+h, x+w//2:x+w]
            scan_subregions(left_half, x, y, w//2, h)
            scan_subregions(right_half, x+w//2, y, w//2, h)
        else:  # Vertical shape, split horizontally
            top_half = mask[y:y+h//2, x:x+w]
            bottom_half = mask[y+h//2:y+h, x:x+w]
            scan_subregions(top_half, x, y, w, h//2)
            scan_subregions(bottom_half, x, y+h//2, w, h//2)
    else:
        cv2.circle(frame_inv, center, r, (0, 255, 0), 2)
        cv2.circle(mask_circles, center, r, (255, 255, 255), -1)

# Apply mask to retain only enclosed areas
frame_result = cv2.bitwise_and(frame_inv, mask_circles)

cv2.imshow('Detected Circles', frame_result)
cv2.waitKey(0)
cv2.destroyAllWindows()