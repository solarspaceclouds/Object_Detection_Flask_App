import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import uuid

app = Flask(__name__)

# Set up paths for YOLOv3 model and class names
yolo_weights = "yolov3.weights"
yolo_cfg = "yolov3.cfg"
coco_names = "coco.names"

# Create a directory to store result images
result_image_dir = "result_images"
os.makedirs(result_image_dir, exist_ok=True)

# Load YOLOv3 model and class names
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
with open(coco_names, "r") as f:
    classes = f.read().strip().split("\n")

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to get a unique color for each object type
def get_object_color(obj_type):
    color_mapping = {
        "car": (0, 0, 255),       # Red for cars
        "bus": (0, 255, 0),       # Green for buses
        "truck": (255, 0, 0),     # Blue for trucks
        "motorbike": (255, 255, 0) # Yellow for motorbikes
    }
    return color_mapping.get(obj_type, (0, 255, 255)) # Default to cyan for other types

# Function to perform object detection
def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Prepare image for YOLOv3
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform detection
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    results = []
    car_count = 0
    motorbike_count = 0
    bus_count = 0
    truck_count = 0

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                class_name = classes[class_id]

                if class_name in ["car", "motorbike", "bus", "truck"]:
                    x, y, w, h = obj[0:4] * np.array([width, height, width, height])
                    x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

                    iou_threshold = 0.5  # Set your specific IoU threshold here

                    # Check if there is significant overlap with any existing item in results
                    is_overlap = False
                    for item in results:
                        iou = calculate_iou((x, y, x + w, y + h), item)
                        if iou > iou_threshold:
                            is_overlap = True
                            break

                    if not is_overlap:
                        results.append((x, y, x + w, y + h))

                        color = get_object_color(class_name)

                        if class_name == "car":
                            car_count += 1
                        elif class_name == "motorbike":
                            motorbike_count += 1
                        elif class_name == "bus":
                            bus_count += 1
                        elif class_name == "truck":
                            truck_count += 1

                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    print("Results:", results)
    print("Number of Cars Detected:", car_count)
    print("Number of Motorbikes Detected:", motorbike_count)
    print("Number of Buses Detected:", bus_count)
    print("Number of Trucks Detected:", truck_count)

    return image, car_count, motorbike_count, bus_count, truck_count

@app.route('/')
def index():
    return render_template('index_vehicle_det.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    
    if image.filename == '':
        return redirect(request.url)

    # Save the uploaded image to a temporary location
    image_path = 'temp_image.jpg'
    image.save(image_path)

    # Perform object detection based on the specified object_type
    detected_image, car_count, motorbike_count, bus_count, truck_count = detect_objects(image_path)
    num_objects_detected = car_count + motorbike_count + bus_count + truck_count  # Total number of detected objects

    # Generate a unique filename for the result image
    result_image_filename = f"result_{uuid.uuid4().hex}.jpg"
    result_image_path = os.path.join(result_image_dir, result_image_filename)

    # Save the image with bounding boxes
    cv2.imwrite(result_image_path, detected_image)

    return render_template('result_vehicle_det.html', result_image=result_image_filename,num_cars= car_count, num_motorbikes= motorbike_count, num_buses = bus_count, num_trucks = truck_count, total_objects=num_objects_detected)

@app.route('/results/<result_image_filename>')
def results(result_image_filename):
    return send_from_directory(result_image_dir, result_image_filename)

if __name__ == '__main__':
    app.run(debug=True)
