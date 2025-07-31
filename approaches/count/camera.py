from ultralytics import YOLO
import cv2

# Use the pretrained YOLOv8 model (COCO dataset)
model = YOLO('yolov8n.pt')  # yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.

# img = cv2.imread('test-data/image.webp')
img = cv2.imread('test-data/chickens-.jpg')
results = model(img)

# Filter detections for "bird" class (COCO: class index 14)
bird_class_index = 14
bird_boxes = [box.xyxy for box in results[0].boxes if int(box.cls) == bird_class_index]

print(bird_boxes)
# counter = bird_boxes.count()

for box in bird_boxes:
    x1, y1, x2, y2 = map(int, box[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, 'chick', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# cv2.putText(img, 'str(counter)', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow('output.jpg', img)
cv2.waitKey(0)
print(f"Detected {len(bird_boxes)} birds. Output saved to output.jpg.")