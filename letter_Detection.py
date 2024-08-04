
import os
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

image_path = 'static/devanagariImg.PNG'
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use pytesseract to detect words and their bounding boxes
boxes = pytesseract.image_to_boxes(gray)

output_dir = 'output2_boxes'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def resize_and_normalize_image(image, target_size=(32, 32)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image while preserving aspect ratio
    height, width = image.shape[:2]
    scale = min(target_size[0] / height, target_size[1] / width)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))  

    if scale > 1:  
        interpolation_method = cv2.INTER_CUBIC  
    else:  
        interpolation_method = cv2.INTER_AREA
    
    resized_image = cv2.resize(image, new_size, interpolation_method)
    
    # Create a target image with the desired size
    target_image = np.zeros(target_size, dtype=np.uint8)
    y_offset = (target_size[0] - new_size[1]) // 2
    x_offset = (target_size[1] - new_size[0]) // 2
    target_image[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized_image
    
    # Normalize the image
    normalized_image = target_image / 255.0
    
    return normalized_image

# Loop through each box
for i, b in enumerate(boxes.splitlines()):
    b = b.split()
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    
    cropped_image = image[image.shape[0] - h:image.shape[0] - y, x:w]
    
    if cropped_image.size == 0:
        print(f"Skipping empty region for box {i}")
        continue
    
    processed_image = resize_and_normalize_image(cropped_image, target_size=(32, 32))
    
    # Convert to 8-bit image for saving
    final_image = (processed_image * 255).astype(np.uint8)
    
    box_image_path = os.path.join(output_dir, f'box_{i}.jpg')
    cv2.imwrite(box_image_path, final_image)
    
    # Draw bounding boxes on the original image
    cv2.rectangle(image, (x, image.shape[0] - y), (w, image.shape[0] - h), (255, 0, 0), 3)

cv2.imwrite('devanagari_img_box.jpg', image)

# Display the processed image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

