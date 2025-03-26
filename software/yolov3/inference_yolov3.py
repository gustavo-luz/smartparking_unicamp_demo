# prev version working but not exactly as expected as the mask needs to be resized also

import cv2
import os
import csv
import numpy as np
from pytorchyolo import detect, models
from PIL import Image
import time
import os
import psutil
from datetime import datetime


"""CFG START"""
model_path = "yolov3.cfg"
weights_path = "yolov3.weights"
model = models.load_model(model_path, weights_path)
# image_dir = '../../assets/demo_images'
# output_dir = '../../assets/results/results_yolov3/yolov3'
image_dir = '../../assets/original'
output_dir = '../../assets/results/results_yolov3_ic2/yolov3_ic2'
savefigs = 'debug' #choose 'no' to not save images and 'debug' to save images
"""CFG END"""

now = datetime.now()
filename_timestamp = now.strftime("%Y%m%dT%H%M%S")
output_path = f'{output_dir}/batch_{filename_timestamp}'
output_csv_path = output_path
OUTPUT_FILE = f'{output_path}/df_individual_metrics_{filename_timestamp}.csv' 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_csv_path, exist_ok=True)



# Define the class ID for "car" (COCO dataset)
CAR_CLASS_ID = 2


with Image.open('mask_original_img_768_1024_bw.png') as mask:
    # mask = mask.resize((640, 640))
    mask = np.array(mask)


# Non-Maximum Suppression (NMS) function
def nms(boxes, scores, iou_threshold=0.5):
    x = boxes[:, 0]
    y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    areas = width * height
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + width[i], x[order[1:]] + width[order[1:]])
        yy2 = np.minimum(y[i] + height[i], y[order[1:]] + height[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def detection_matrix_modified(x,y,mask, img_width, img_height):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Use the first channel if it's multi-channel
    
    # Get original mask dimensions
    mask_height, mask_width = mask.shape[:2]
    image_size = 416  # Image is resized to 640x640

    scale_x =  mask_height / image_size
    scale_y = mask_width / image_size

    # scale_x =  mask_width / image_size
    # scale_y = mask_height / image_size

    x_mask = int(x * scale_x)
    y_mask = int(y * scale_y)

    # print(f"\n points are {x},{y} \n mask shape is {mask.shape}\n mask_x = {x_mask}, mask_y = {y_mask}")
    pixel_value = mask[int(y_mask),int(x_mask)]#mask[int(y),int(x)]
    # pixel_value = mask[int(x_mask),int(y_mask)]#mask[int(y),int(x)]
    print(f"\n points are {x},{y} \n pixel value: {pixel_value} and mask \n mask_x = {x_mask}, mask_y = {y_mask}")
    if pixel_value == 255:
        print("The point is outside the mask.")
        return False
    else:
        print("The point is inside the mask.")
        return True

# Function to count cars inside the mask
def count_cars_post(boxes, mask, img_width, img_height):
    car_count = 0
    mask_x, mask_y = mask.shape[:2]
    for box in boxes:

        box_ = np.array(box)
        x_center, y_center, width, height = box_

        point_inside = detection_matrix_modified(x_center, y_center, mask,416, 416)
        if point_inside:
            car_count += 1
    return car_count

# Open the CSV file for writing
with open(OUTPUT_FILE, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(['image_name','predicted_cars', 'processing_time','cpu_usage', 'memory_used', 'swap_used'])

    # Loop through every image in the directory
    for image_name in os.listdir(image_dir):
        # Construct the full image path
        image_path = os.path.join(image_dir, image_name)

        print(f"image path is {image_path}")
        with Image.open(image_path) as img:
            img = img.resize((416, 416))

            img = np.array(img)            

            # Skip if the image is not loaded successfully
            if img is None:
                print(f"Error: Unable to load image at {image_path}")
                continue

            # Get image dimensions
            img_height, img_width, _ = img.shape

            # Convert OpenCV BGR to RGB (YOLO expects RGB images)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run the YOLO model on the image
            start_time = time.time()
            boxes = detect.detect_image(model, img_rgb)

            # Filter for cars and extract bounding boxes, scores, and class IDs
            car_boxes = []
            scores = []
            for box in boxes:
                print(f'box: {box}')
                x1, y1, x2, y2, confidence, class_id = box
                # if int(class_id) == CAR_CLASS_ID:  # Check if the detected object is a car
                if (int(class_id) == 2 or int(class_id) == 7):
                    car_boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)

            # Apply Non-Maximum Suppression (NMS)
            if car_boxes:
                car_boxes = np.array(car_boxes)
                scores =  car_boxes[:, 3]
                keep = nms(car_boxes, scores, iou_threshold=0.5)
                car_boxes = car_boxes[keep]

                # Count cars inside the mask
                car_count = count_cars_post(car_boxes, mask, img_width, img_height)
            else:
                car_count = 0

            # count nms time
            inference_time = time.time() - start_time

            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()
            memory_used = memory_info.used / (1024 ** 2)  # Convert to MB
            swap_used = swap_info.used / (1024 ** 2)  # Convert to MB
            # Write the image name and number of cars to the CSV file
            
            csv_writer.writerow([image_name, car_count,inference_time, cpu_usage, memory_used, swap_used])
            if savefigs == 'debug':
                # Draw bounding boxes on the image
                for box in car_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"Car: {confidence:.2f}"
                    cv2.putText(img_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add the total number of cars detected to the image
                text = f"Total Cars: {car_count}"
                cv2.putText(img_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                # Optionally, save the output image with bounding boxes
                output_image_path = os.path.join(output_csv_path, f"output_{image_name}")
                cv2.imwrite(output_image_path, img_rgb)

            # Print the number of cars detected for the current image
            print(f"Image: {image_name}, Number of cars detected: {car_count}")

print(f"Predictions saved to {output_csv_path}")