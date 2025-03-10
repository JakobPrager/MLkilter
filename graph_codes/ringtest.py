import cv2
import numpy as np

import cv2

# Load the image
image = cv2.imread("test/frame_0000.jpg")

# Get original dimensions
height, width = image.shape[:2]

# Define scale factor
scale_factor = 1/0.788

# Compute new dimensions
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Save the resized image (optional)
cv2.imwrite("test/rez_0000.jpg", resized_image)

# Display the resized image
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Grid size
rows, cols = 19, 19

# Image parameters
ring_radius = 20 # Radius of each ring
thickness = ring_radius // 5  # Thickness of the ring
spacing = 2 * ring_radius   # Spacing between rings

# Calculate image size
img_height = (rows+1) * spacing + spacing + spacing + spacing
img_width = cols * spacing  

all_centers = []
# Create a white canvas
#image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
image = cv2.imread("Blank_align.jpg")
offset = 20
off_up = 20
# Draw the rings
for i in range(rows):
    for j in np.arange(1,cols-1):
        center_x = j * spacing + spacing // 2 - offset
        center_y = i * spacing + spacing // 2 + spacing*2 - offset - off_up
        all_centers.append((center_x, center_y))
        cv2.circle(image, (center_x, center_y), ring_radius, (255, 155, 155), thickness)

# Draw middle rings in a checkerboard pattern
middle_ring_centers = []
for i in range(2, rows - 1, 2):  # Every second row
    for j in range(0, cols - 1, 2):  # Every second column
        center_x = j * spacing + spacing - offset
        center_y = i * spacing + spacing + spacing*2 - offset - off_up
        #change color to orange
        all_centers.append((center_x, center_y))
        cv2.circle(image, (center_x, center_y), ring_radius, (255,255,255), thickness= thickness)
        middle_ring_centers.append((center_x, center_y))

# Draw a ring in the middle of each 4 middle rings
for i in range(0, len(middle_ring_centers) - (cols // 2), cols // 2):
    for j in range(0, cols // 2 - 1):
        idx = i + j
        if idx + 1 < len(middle_ring_centers) and idx + (cols // 2) < len(middle_ring_centers):
            cx1, cy1 = middle_ring_centers[idx]
            cx2, cy2 = middle_ring_centers[idx + 1]
            cx3, cy3 = middle_ring_centers[idx + (cols // 2)]
            cx4, cy4 = middle_ring_centers[idx + (cols // 2) + 1]
            center_x = (cx1 + cx4) // 2 
            center_y = (cy1 + cy4) // 2 
            all_centers.append((center_x, center_y))
            cv2.circle(image, (center_x, center_y), ring_radius, (255, 200, 155), thickness)
            
# Add an extra half row of rings at the very bottom
for j in range(0, cols - 1):
    center_x = j * spacing + ring_radius  +spacing // 2 - offset
    center_y = rows * spacing + spacing *2  - offset- off_up# Extra row at bottom
    all_centers.append((center_x, center_y))
    cv2.circle(image, (center_x, center_y), ring_radius, (255, 255, 50), thickness)

# Add an extra half row of rings at the very bottom
for j in range(3, rows - 2,2):
    center_x = (cols-1) * spacing - offset
    center_y = j * spacing + spacing  + spacing*2 -offset -off_up# Extra row at bottom
    all_centers.append((center_x, center_y))
    cv2.circle(image, (center_x, center_y), ring_radius, (255, 255, 50), thickness)

# Show the image
cv2.imshow("Ring Grid", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(all_centers)
#save all centers in a csv file
import csv
with open('ring_centers.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['x','y'])
    for center in all_centers:
        writer.writerow(center)

## Save the image
#cv2.imwrite("ring_grid3.png", image)