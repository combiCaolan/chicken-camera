#!/usr/bin/env python3
"""
Quick YOLO v11 Object Detection on Image
"""

import cv2
from ultralytics import YOLO

# Configuration
IMAGE_PATH = 'test-data/image.webp'  # Change this to your image path
MODEL_SIZE = 'yolo11n.pt'      # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
CONFIDENCE = 0.1               # Confidence threshold
SAVE_OUTPUT = True             # Save annotated image
OUTPUT_PATH = 'detected_output.jpg'

def main():
    # Load YOLO model
    print(f"Loading {MODEL_SIZE}...")
    model = YOLO(MODEL_SIZE)
    
    # Load image
    print(f"Loading image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image {IMAGE_PATH}")
        return
    
    # Run detection
    print("Running detection...")
    results = model(image, conf=CONFIDENCE)
    
    # Process results
    for result in results:
        # Get annotated image
        annotated_image = result.plot()
        
        # Print detections
        if result.boxes is not None:
            print(f"\nDetected {len(result.boxes)} objects:")
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                print(f"  - {class_name}: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        else:
            print("No objects detected")
        
        # Show image
        cv2.imshow('YOLO v11 Detection', annotated_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if requested
        if SAVE_OUTPUT:
            cv2.imwrite(OUTPUT_PATH, annotated_image)
            print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 