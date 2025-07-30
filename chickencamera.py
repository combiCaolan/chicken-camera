# importing libraries
import cv2
import time
import os
from ultralytics import YOLO
import numpy as np
import math
import matplotlib.pyplot as plt

# Configuration - Optimized for chicken detection
USE_YOLO12 = True  # YOLO11 is faster
CONFIDENCE_THRESHOLD = 0.09  # Increased for better reliability
SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = 'chicken_detection_output.mp4'
FRAME_SKIP = 1  # Process every frame for better tracking
TextShow = False  # Show text overlays on video

# Video source
VIDEO_SOURCE = 'test-data/RECORDING.mp4'

# VIDEO_SOURCE = 'test-data/YELLOW-recording.mp4'


# ROI (Region of Interest) Configuration
USE_ROI = True  # Set to False to analyze full frame
# ROI coordinates as percentage of frame (x1, y1, x2, y2) - 0.0 to 1.0
ROI_COORDS = (0.005509641873278237, 0.00966183574879227, 0.4022038567493113, 0.5458937198067633)

# Counting line configuration - will be set interactively
COUNTING_LINE_POINT1 = (50, 200)  # Default first point
COUNTING_LINE_POINT2 = (400, 150)  # Default second point

# Model download directory
MODEL_DIR = "models"

# Classes that chickens might be detected as - focusing on 'bird' for reliability
POULTRY_CLASSES = ['bird']  # Simplified to just birds for better accuracy

# Simplified tracking system
class ChickenTracker:
    def __init__(self):
        self.tracks = {}  # {track_id: {'positions': [(x,y), ...], 'last_seen': frame_num, 'crossed': bool}}
        self.next_id = 1
        self.crossing_count = 0
        self.current_frame = 0
        self.max_distance = 80  # Maximum distance to associate detection with existing track
        self.max_missing_frames = 10  # Remove tracks after this many missing frames
        
    def update(self, detections, line_point1, line_point2):
        """Update tracker with new detections"""
        self.current_frame += 1
        
        # Get chicken centers from detections
        current_detections = []
        for detection in detections:
            if detection['name'] not in POULTRY_CLASSES:
                print('Skipping non-poultry detection:', detection['name'])
            if detection['name'] in POULTRY_CLASSES:
                x1, y1, x2, y2 = detection['box']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_detections.append(center)
        
        # Match detections to existing tracks
        matched_tracks = set()
        
        for center in current_detections:
            best_track_id = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_data in self.tracks.items():
                if len(track_data['positions']) > 0:
                    last_pos = track_data['positions'][-1]
                    distance = math.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                    
                    if distance < self.max_distance and distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
            
            if best_track_id is not None and best_track_id not in matched_tracks:
                # Update existing track
                self.tracks[best_track_id]['positions'].append(center)
                self.tracks[best_track_id]['last_seen'] = self.current_frame
                matched_tracks.add(best_track_id)
                
                # Check for line crossing
                if len(self.tracks[best_track_id]['positions']) >= 2 and not self.tracks[best_track_id]['crossed']:
                    if self._check_line_crossing(
                        self.tracks[best_track_id]['positions'][-2], 
                        self.tracks[best_track_id]['positions'][-1], 
                        line_point1, line_point2
                    ):
                        self.tracks[best_track_id]['crossed'] = True
                        self.crossing_count += 1
                        print(f"🐔 CHICKEN #{self.crossing_count} CROSSED THE LINE! (Track ID: {best_track_id})")
                        
            else:
                # Create new track
                self.tracks[self.next_id] = {
                    'positions': [center],
                    'last_seen': self.current_frame,
                    'crossed': False
                }
                self.next_id += 1
        
        # Clean up old tracks
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            if self.current_frame - track_data['last_seen'] > self.max_missing_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Limit track history to prevent memory issues
        for track_data in self.tracks.values():
            if len(track_data['positions']) > 20:
                track_data['positions'] = track_data['positions'][-20:]
        
        return self.crossing_count
    
    def _check_line_crossing(self, prev_pos, curr_pos, line_point1, line_point2):
        """Check if movement from prev_pos to curr_pos crosses the line"""
        x1, y1 = line_point1
        x2, y2 = line_point2
        
        # Calculate which side of line each point is on
        def point_side(px, py):
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        prev_side = point_side(prev_pos[0], prev_pos[1])
        curr_side = point_side(curr_pos[0], curr_pos[1])
        
        # Crossing occurs if signs are different (and both are significant)
        return (prev_side > 5 and curr_side < -5) or (prev_side < -5 and curr_side > 5)
    
    def draw_tracks(self, frame):
        """Draw tracking visualization on frame"""
        for track_id, track_data in self.tracks.items():
            positions = track_data['positions']
            
            # Draw track trail
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    color = (0, 255, 0) if track_data['crossed'] else (128, 128, 128)
                    cv2.line(frame, positions[i-1], positions[i], color, 2)
            
            # Draw current position and ID
            if len(positions) > 0:
                pos = positions[-1]
                color = (0, 255, 0) if track_data['crossed'] else (255, 255, 255)
                cv2.circle(frame, pos, 5, color, -1)
                if TextShow:
                    cv2.putText(frame, str(track_id), (pos[0]-10, pos[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Initialize tracker
tracker = ChickenTracker()

def ensure_model_downloaded():
    """Download and cache YOLO models locally for faster loading"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if USE_YOLO12:
        model_name = 'yolo12n.pt'
        model_path = os.path.join(MODEL_DIR, model_name)
        display_name = "YOLO12n"
    else:
        model_name = 'yolo11n.pt'
        model_path = os.path.join(MODEL_DIR, model_name)
        display_name = "YOLO11n"
    
    if os.path.exists(model_path):
        print(f"Loading cached model from: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"Downloading {model_name} for first time...")
        model = YOLO(model_name)
        
        try:
            import shutil
            default_path = model.ckpt_path if hasattr(model, 'ckpt_path') else None
            if default_path and os.path.exists(default_path):
                shutil.copy2(default_path, model_path)
                print(f"Model cached to: {model_path}")
        except Exception as e:
            print(f"Note: Could not cache model locally: {e}")
    
    print(f"Model loaded: {display_name}")
    print(f"Available classes: {len(model.names)}")
    
    # Show relevant classes
    print("Relevant classes for poultry detection:")
    for class_id, class_name in model.names.items():
        if class_name in POULTRY_CLASSES:
            print(f"  {class_id}: {class_name}")
    
    return model, display_name

def setup_roi_selector(frame):
    """Interactive ROI selection - click and drag to select region"""
    global ROI_COORDS, USE_ROI
    
    print("\n" + "="*50)
    print("ROI SELECTION MODE")
    print("="*50)
    print("Instructions:")
    print("- Click and drag to select the region where chickens are located")
    print("- Press SPACE to confirm selection")
    print("- Press 'f' to use full frame (no ROI)")
    print("- Press 'r' to reset and try again")
    print("- Current ROI covers bottom 70% of frame by default")
    
    roi_frame = frame.copy()
    
    # Draw current ROI
    h, w = frame.shape[:2]
    x1 = int(ROI_COORDS[0] * w)
    y1 = int(ROI_COORDS[1] * h)
    x2 = int(ROI_COORDS[2] * w)
    y2 = int(ROI_COORDS[3] * h)
    
    cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if TextShow:
        cv2.putText(roi_frame, 'Current ROI (green box)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(roi_frame, 'Drag to select new ROI, SPACE to confirm, F for full frame', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mouse callback for ROI selection
    drawing = False
    start_point = None
    current_roi = list(ROI_COORDS)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, current_roi, roi_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, start_point, (x, y), (255, 0, 0), 2)
            if TextShow:
                cv2.putText(temp_frame, 'Drag to select ROI, SPACE to confirm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('ROI Selection', temp_frame)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if start_point:
                h, w = frame.shape[:2]
                x1_norm = min(start_point[0], x) / w
                y1_norm = min(start_point[1], y) / h
                x2_norm = max(start_point[0], x) / w
                y2_norm = max(start_point[1], y) / h
                
                current_roi = (x1_norm, y1_norm, x2_norm, y2_norm)
                
                roi_frame = frame.copy()
                x1_px = int(current_roi[0] * w)
                y1_px = int(current_roi[1] * h)
                x2_px = int(current_roi[2] * w)
                y2_px = int(current_roi[3] * h)
                cv2.rectangle(roi_frame, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
                cv2.putText(roi_frame, 'Selected ROI (press SPACE to confirm)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('ROI Selection', roi_frame)
    
    cv2.namedWindow('ROI Selection')
    cv2.setMouseCallback('ROI Selection', mouse_callback)
    cv2.imshow('ROI Selection', roi_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            ROI_COORDS = tuple(current_roi)
            USE_ROI = True
            print(f"ROI confirmed: {ROI_COORDS}")
            break
        elif key == ord('f'):
            USE_ROI = False
            print("Using full frame (no ROI)")
            break
        elif key == ord('r'):
            current_roi = list(ROI_COORDS)
            roi_frame = frame.copy()
            h, w = frame.shape[:2]
            x1 = int(ROI_COORDS[0] * w)
            y1 = int(ROI_COORDS[1] * h)
            x2 = int(ROI_COORDS[2] * w)
            y2 = int(ROI_COORDS[3] * h)
            cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(roi_frame, 'Current ROI (green box)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('ROI Selection', roi_frame)
        elif key == ord('q'):
            cv2.destroyWindow('ROI Selection')
            return False
    
    cv2.destroyWindow('ROI Selection')
    return True

def setup_line_selector(frame):
    """Interactive counting line selection - click two points to define the line"""
    global COUNTING_LINE_POINT1, COUNTING_LINE_POINT2
    
    print("\n" + "="*50)
    print("COUNTING LINE SELECTION MODE")
    print("="*50)
    print("Instructions:")
    print("- Click FIRST point where you want the counting line to start")
    print("- Click SECOND point where you want the counting line to end")
    print("- Press SPACE to confirm line selection")
    print("- Press 'r' to reset and try again")
    print("- Press 'ESC' to use default line")
    
    line_frame = frame.copy()
    
    # Draw current line
    cv2.line(line_frame, COUNTING_LINE_POINT1, COUNTING_LINE_POINT2, (0, 0, 255), 4)
    if TextShow:
        cv2.putText(line_frame, 'Current counting line (red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(line_frame, 'Click 2 points to set new line, SPACE to confirm', (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mouse callback for line selection
    points_selected = []
    current_line = [COUNTING_LINE_POINT1, COUNTING_LINE_POINT2]
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points_selected, current_line, line_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points_selected.append((x, y))
            
            # Draw the point
            temp_frame = frame.copy()
            
            # Draw existing points
            for i, point in enumerate(points_selected):
                color = (255, 0, 0) if i == 0 else (0, 255, 255)  # Blue for first, cyan for second
                cv2.circle(temp_frame, point, 8, color, -1)
                cv2.putText(temp_frame, f'Point {i+1}', (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw line if we have 2 points
            if len(points_selected) == 2:
                cv2.line(temp_frame, points_selected[0], points_selected[1], (0, 0, 255), 4)
                cv2.putText(temp_frame, 'New counting line - Press SPACE to confirm', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                current_line = points_selected.copy()
            elif len(points_selected) == 1:
                cv2.putText(temp_frame, f'Click second point (selected: {len(points_selected)}/2)', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(temp_frame, 'Click first point for counting line', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Line Selection', temp_frame)
            line_frame = temp_frame
            
            # Reset if more than 2 points clicked
            if len(points_selected) > 2:
                points_selected = []
    
    cv2.namedWindow('Line Selection')
    cv2.setMouseCallback('Line Selection', mouse_callback)
    cv2.imshow('Line Selection', line_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Confirm selection
            if len(points_selected) == 2:
                COUNTING_LINE_POINT1 = points_selected[0]
                COUNTING_LINE_POINT2 = points_selected[1]
                print(f"Counting line confirmed: {COUNTING_LINE_POINT1} to {COUNTING_LINE_POINT2}")
                break
            else:
                print("Please select exactly 2 points before confirming")
        elif key == ord('r'):  # Reset
            points_selected = []
            line_frame = frame.copy()
            cv2.line(line_frame, COUNTING_LINE_POINT1, COUNTING_LINE_POINT2, (0, 0, 255), 4)
            cv2.putText(line_frame, 'Current counting line (red)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(line_frame, 'Click 2 points to set new line, SPACE to confirm', (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Line Selection', line_frame)
        elif key == 27:  # ESC key - use default
            print("Using default counting line")
            break
        elif key == ord('q'):
            cv2.destroyWindow('Line Selection')
            return False
    
    cv2.destroyWindow('Line Selection')
    return True

def extract_roi(frame, roi_coords):
    """Extract region of interest from frame"""
    if not USE_ROI:
        return frame, (0, 0)
    
    h, w = frame.shape[:2]
    x1 = int(roi_coords[0] * w)
    y1 = int(roi_coords[1] * h)
    x2 = int(roi_coords[2] * w)
    y2 = int(roi_coords[3] * h)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi_frame = frame[y1:y2, x1:x2]
    return roi_frame, (x1, y1)

def setup_video_capture(source):
    """Initialize video capture with optimizations"""
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return None, None, None, None
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"Processing every {FRAME_SKIP} frame(s)")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    return cap, frame_width, frame_height, fps

def darken_frame(frame, darkness_factor=0.7):
    """
    Darken the frame before detection
    darkness_factor: 0.0 = completely black, 1.0 = no change, values > 1.0 = brighter
    """
    # Method 1: Simple multiplication (fastest)
    darkened = cv2.multiply(frame, darkness_factor)
    
    # Method 2: Alternative using addWeighted for more control
    # darkened = cv2.addWeighted(frame, darkness_factor, np.zeros(frame.shape, frame.dtype), 0, 0)
    
    # Method 3: Using HSV to adjust only brightness/value channel
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # hsv[:, :, 2] = hsv[:, :, 2] * darkness_factor  # Adjust V (brightness) channel
    # darkened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return darkened.astype(np.uint8)

def process_frame_with_roi(frame, model, model_name, frame_width, frame_height, frame_number, confidence_threshold):
    """Process frame with improved ROI support and robust tracking"""
    global tracker
    
    # DARKEN THE FRAME BEFORE ANY PROCESSING
    # Adjust darkness_factor as needed: 0.5 = half brightness, 0.3 = quite dark
    darkened_frame = darken_frame(frame, darkness_factor=0.6)
    
    # Extract ROI or use full frame (now using darkened frame)
    roi_frame, (offset_x, offset_y) = extract_roi(darkened_frame, ROI_COORDS)
    
    if roi_frame.size == 0:
        print("Warning: ROI is empty, using full frame")
        roi_frame, (offset_x, offset_y) = darkened_frame, (0, 0)
    
    # Run detection on darkened ROI
    try:
        results = model(roi_frame, conf=confidence_threshold, verbose=False, imgsz=640)
        detections = results[0]
    except Exception as e:
        print(f"Detection error: {e}")
        detections = None
    
    # Process detections (rest of the function remains the same)
    poultry_count = 0
    all_detections = []
    class_counts = {}
    
    if detections is not None and detections.boxes is not None:
        for box in detections.boxes:
            try:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Get coordinates and adjust for ROI offset
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                all_detections.append({
                    'name': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
                
                if class_name in POULTRY_CLASSES:
                    poultry_count += 1
                    
                    # Draw detection box ON THE ORIGINAL FRAME (not darkened)
                    # This way the UI elements remain visible
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
            except Exception as e:
                print(f"Error processing detection: {e}")
                continue
    
    # Use the globally selected counting line points
    line_point1 = COUNTING_LINE_POINT1
    line_point2 = COUNTING_LINE_POINT2
    
    # Update tracking and count line crossings
    try:
        total_crossed = tracker.update(all_detections, line_point1, line_point2)
    except Exception as e:
        print(f"Tracking error: {e}")
        total_crossed = tracker.crossing_count
    
    # Draw ROI boundary
    if USE_ROI:
        h, w = frame.shape[:2]
        roi_x1 = int(ROI_COORDS[0] * w)
        roi_y1 = int(ROI_COORDS[1] * h)
        roi_x2 = int(ROI_COORDS[2] * w)
        roi_y2 = int(ROI_COORDS[3] * h)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)
    
    # Draw counting line using selected points
    cv2.line(frame, line_point1, line_point2, (0, 0, 255), 4)
    cv2.putText(frame, 'COUNTING LINE', (line_point1[0], line_point1[1]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw line endpoints for reference
    cv2.circle(frame, line_point1, 6, (255, 0, 0), -1)  # Blue circle at start
    cv2.circle(frame, line_point2, 6, (0, 255, 255), -1)  # Cyan circle at end
    
    # Draw tracking visualization
    tracker.draw_tracks(frame)
    
    # Text overlays
    cv2.putText(frame, f'VISIBLE: {poultry_count}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.putText(frame, f'CROSSED: {total_crossed}', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.putText(frame, f'TRACKS: {len(tracker.tracks)}', (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    roi_text = f"ROI: {'ON' if USE_ROI else 'OFF'}"
    cv2.putText(frame, roi_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Detection breakdown
    y_offset = 170
    for class_name, count in class_counts.items():
        if class_name in POULTRY_CLASSES:
            cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
    
    cv2.putText(frame, f'{model_name} | Conf: {confidence_threshold}', 
               (10, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(frame, f'Frame: {frame_number}', 
               (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add darkening indicator
    cv2.putText(frame, 'DARKENED FOR DETECTION', 
               (frame_width - 250, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    return frame, poultry_count, all_detections, class_counts

def main():
    """Main function with robust error handling and interactive line selection"""
    global tracker  # Move global declaration to top of function
    
    print("="*60)
    print("INTERACTIVE CHICKEN LINE CROSSING COUNTER")
    print("="*60)
    
    # Download/load model
    try:
        model, model_name = ensure_model_downloaded()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Setup video
    cap, frame_width, frame_height, fps = setup_video_capture(VIDEO_SOURCE)
    if cap is None:
        return
    
    # Get first frame for ROI and line selection
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame")
        return
    
    # ROI selection
    if not setup_roi_selector(first_frame):
        print("ROI selection cancelled")
        return
    
    # Interactive line selection after ROI
    if not setup_line_selector(first_frame):
        print("Line selection cancelled")
        return
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Setup video writer
    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = fps // FRAME_SKIP
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (frame_width, frame_height))
        print(f"Output video: {OUTPUT_VIDEO_PATH} @ {output_fps} FPS")
    else:
        out = None
    
    # Processing variables
    frame_number = 0
    processed_frames = 0
    start_time = time.time()
    process_times = []
    confidence_threshold = CONFIDENCE_THRESHOLD
    
    print(f"\nProcessing started...")
    print(f"Settings: Confidence={confidence_threshold}, Frame skip={FRAME_SKIP}, ROI={'ON' if USE_ROI else 'OFF'}")
    print(f"Counting line: {COUNTING_LINE_POINT1} to {COUNTING_LINE_POINT2}")
    print("Controls: 'q'=quit, 'space'=pause, 'c'=adjust confidence, 'r'=reselect ROI, 'l'=reselect line")
    print("🐔 Interactive line tracking - watching for line crossings...")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                if frame_number % FRAME_SKIP != 0:
                    continue
                
                processed_frames += 1
                frame_start = time.time()
                
                # Process frame with error handling
                try:
                    processed_frame, poultry_count, detections, class_counts = process_frame_with_roi(
                        frame, model, model_name, frame_width, frame_height, frame_number, confidence_threshold
                    )
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    processed_frame = frame
                    poultry_count = 0
                    class_counts = {}
                
                frame_time = time.time() - frame_start
                process_times.append(frame_time)
                
                # FPS calculation
                if process_times:
                    avg_time = sum(process_times[-10:]) / len(process_times[-10:])
                    processing_fps = 1.0 / avg_time if avg_time > 0 else 0
                    cv2.putText(processed_frame, f'FPS: {processing_fps:.1f}', 
                               (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if out:
                    out.write(processed_frame)
                
                #Plot the original image
                # plt.subplot(1, 2, 1)
                # plt.title("Original")
                # plt.imshow(image)

                # Adjust the brightness and contrast
                # Adjusts the brightness by adding 10 to each pixel value
                # brightness = 0.001 
                # # Adjusts the contrast by scaling the pixel values by 2.3
                # contrast = 1.3  
                # image2 = cv2.addWeighted(processed_frame, contrast, np.zeros(processed_frame.shape, processed_frame.dtype), 0, brightness)

                
                # Convert the image from BGR to HSV color space
                image = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2HSV)

                # Adjust the hue, saturation, and value of the image
                # Adjusts the hue by multiplying it by 0.7
                image[:, :, 0] = image[:, :, 0] * 0.7
                # Adjusts the saturation by multiplying it by 1.5
                image[:, :, 1] = image[:, :, 1] * 1.5
                # Adjusts the value by multiplying it by 0.5
                image[:, :, 2] = image[:, :, 2] * 0.5

                # Convert the image back to BGR color space
                image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


                #Save the image
                # cv2.imwrite('modified_image.jpg', image2)
                #Plot the contrast image
                # plt.subplot(1, 2, 2)
                # plt.title("Brightness & contrast")
                # plt.imshow(image2)
                # plt.show()

                # cv2.imshow('Interactive Chicken Counter', processed_frame)
                cv2.imshow('Interactive Chicken Counter', image2)
                
                # Progress reporting
                if processed_frames % 30 == 0:  # Less frequent to reduce spam
                    print(f"Frame {processed_frames}: {poultry_count} visible | {tracker.crossing_count} crossed | {len(tracker.tracks)} active tracks")
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF  # Faster response
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('c'):
                print(f"Current confidence: {confidence_threshold}")
                try:
                    new_conf = float(input("New confidence (0.1-0.9): "))
                    if 0.1 <= new_conf <= 0.9:
                        confidence_threshold = new_conf
                        print(f"Updated to: {confidence_threshold}")
                except ValueError:
                    print("Invalid input")
            elif key == ord('r'):  # Reselect ROI
                paused = True
                if setup_roi_selector(frame):
                    print("ROI updated")
                paused = False
            elif key == ord('l'):  # Reselect counting line
                paused = True
                if setup_line_selector(frame):
                    print("Counting line updated")
                    # Reset tracker when line changes to avoid confusion
                    tracker = ChickenTracker()
                    print("Tracker reset due to line change")
                paused = False
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final stats
        total_time = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"Processed {processed_frames} frames in {total_time:.2f}s")
        print(f"Average FPS: {processed_frames/total_time:.2f}")
        print(f"🐔 FINAL COUNT: {tracker.crossing_count} chickens crossed the line!")
        print(f"Final counting line was: {COUNTING_LINE_POINT1} to {COUNTING_LINE_POINT2}")
        if SAVE_OUTPUT_VIDEO:
            print(f"Saved: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()