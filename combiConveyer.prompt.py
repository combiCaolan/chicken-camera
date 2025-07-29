import cv2
import numpy as np
import time
from collections import deque
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class ChickenCounterConfig:
    """Configuration for different detection methods"""
    
    # Method selection - try different approaches
    DETECTION_METHOD = 'background_subtraction'  # Options: 'background_subtraction', 'motion_detection', 'contour_analysis', 'blob_detection', 'template_matching'
    
    # Video settings
    FRAME_SKIP = 1  # Process every frame for accuracy
    
    # Conveyor settings
    BELT_DIRECTION = 'vertical'  # 'horizontal' or 'vertical'
    COUNTING_LINE_POSITION = 0.6  # 60% down the frame
    
    # Size constraints (adjust based on your chicken size from overhead)
    MIN_CHICKEN_AREA = 800    # Minimum area in pixels
    MAX_CHICKEN_AREA = 8000   # Maximum area in pixels
    MIN_CHICKEN_WIDTH = 20    # Minimum width
    MAX_CHICKEN_WIDTH = 120   # Maximum width
    MIN_CHICKEN_HEIGHT = 20   # Minimum height
    MAX_CHICKEN_HEIGHT = 120  # Maximum height
    
    # Background subtraction settings
    BACKGROUND_LEARNING_RATE = 0.01
    BACKGROUND_THRESHOLD = 30
    
    # Motion detection settings
    MOTION_THRESHOLD = 25
    BLUR_SIZE = 5
    
    # Morphological operations
    ERODE_ITERATIONS = 2
    DILATE_ITERATIONS = 3
    
    # Contour filtering
    MIN_CONTOUR_AREA = 500
    MAX_CONTOUR_AREA = 10000
    
    # Blob detection parameters
    BLOB_MIN_THRESHOLD = 50
    BLOB_MAX_THRESHOLD = 200
    BLOB_MIN_AREA = 500
    BLOB_MAX_AREA = 8000
    
    # Output settings
    SAVE_OUTPUT = True
    OUTPUT_PATH = 'chicken_count_alternative.mp4'
    DEBUG_MODE = True  # Show intermediate processing steps

class AlternativeChickenCounter:
    """Multiple computer vision approaches for chicken counting"""
    
    def __init__(self, config):
        self.config = config
        self.method = config.DETECTION_METHOD
        
        # Counting variables
        self.total_count = 0
        self.frame_number = 0
        
        # Method-specific initialization
        self.background_subtractor = None
        self.previous_frame = None
        self.chicken_template = None
        self.blob_detector = None
        
        # Crossing detection
        self.crossed_objects = set()
        self.object_tracks = {}
        self.next_object_id = 1
        
        self.initialize_method()
    
    def initialize_method(self):
        """Initialize the selected detection method"""
        print(f"Initializing {self.method} detection method...")
        
        if self.method == 'background_subtraction':
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True,
                varThreshold=self.config.BACKGROUND_THRESHOLD
            )
        
        elif self.method == 'blob_detection':
            # Setup blob detector
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = self.config.BLOB_MIN_THRESHOLD
            params.maxThreshold = self.config.BLOB_MAX_THRESHOLD
            params.filterByArea = True
            params.minArea = self.config.BLOB_MIN_AREA
            params.maxArea = self.config.BLOB_MAX_AREA
            params.filterByCircularity = False
            params.filterByConvexity = False
            params.filterByInertia = False
            
            self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def detect_chickens_background_subtraction(self, frame):
        """Detect chickens using background subtraction"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame, learningRate=self.config.BACKGROUND_LEARNING_RATE)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.MIN_CONTOUR_AREA < area < self.config.MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Size filtering
                if (self.config.MIN_CHICKEN_WIDTH < w < self.config.MAX_CHICKEN_WIDTH and
                    self.config.MIN_CHICKEN_HEIGHT < h < self.config.MAX_CHICKEN_HEIGHT):
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area,
                        'contour': contour
                    })
        
        return detections, fg_mask
    
    def detect_chickens_motion_detection(self, frame):
        """Detect chickens using frame differencing and motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.config.BLUR_SIZE, self.config.BLUR_SIZE), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return [], np.zeros_like(gray)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(frame_diff, self.config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=self.config.ERODE_ITERATIONS)
        thresh = cv2.dilate(thresh, kernel, iterations=self.config.DILATE_ITERATIONS)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.MIN_CHICKEN_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                
                if (self.config.MIN_CHICKEN_WIDTH < w < self.config.MAX_CHICKEN_WIDTH and
                    self.config.MIN_CHICKEN_HEIGHT < h < self.config.MAX_CHICKEN_HEIGHT):
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area,
                        'contour': contour
                    })
        
        self.previous_frame = gray
        return detections, thresh
    
    def detect_chickens_contour_analysis(self, frame):
        """Detect chickens using advanced contour analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine thresholding and edge detection
        combined = cv2.bitwise_or(adaptive_thresh, edges)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.MIN_CHICKEN_AREA:
                # Calculate shape properties
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                extent = float(area) / (w * h)
                
                # Filter based on shape characteristics
                if (0.5 < aspect_ratio < 2.0 and  # Not too elongated
                    extent > 0.3 and  # Reasonable fill ratio
                    self.config.MIN_CHICKEN_WIDTH < w < self.config.MAX_CHICKEN_WIDTH and
                    self.config.MIN_CHICKEN_HEIGHT < h < self.config.MAX_CHICKEN_HEIGHT):
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area,
                        'contour': contour,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent
                    })
        
        return detections, combined
    
    def detect_chickens_blob_detection(self, frame):
        """Detect chickens using blob detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Invert image for blob detection (assuming chickens are darker than background)
        inverted = cv2.bitwise_not(gray)
        
        # Detect blobs
        keypoints = self.blob_detector.detect(inverted)
        
        detections = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            # Create bounding box around blob
            w = h = size
            x = max(0, x - w//2)
            y = max(0, y - h//2)
            
            detections.append({
                'bbox': (x, y, w, h),
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'area': size * size,
                'keypoint': kp
            })
        
        # Create visualization mask
        mask = np.zeros_like(gray)
        for detection in detections:
            x, y, w, h = detection['bbox']
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        
        return detections, mask
    
    def track_and_count(self, detections, frame_shape):
        """Track detections across frames and count crossings"""
        h, w = frame_shape[:2]
        
        if self.config.BELT_DIRECTION == 'horizontal':
            counting_line = int(w * self.config.COUNTING_LINE_POSITION)
            line_coord = 0  # x-coordinate
        else:  # vertical
            counting_line = int(h * self.config.COUNTING_LINE_POSITION)
            line_coord = 1  # y-coordinate
        
        # Simple crossing detection based on center positions
        new_counts = 0
        
        for detection in detections:
            center = detection['center']
            detection_id = f"{center[0]}_{center[1]}_{self.frame_number}"
            
            # Check if object crossed the counting line
            if line_coord == 0:  # horizontal belt, check x-coordinate
                if center[0] > counting_line and detection_id not in self.crossed_objects:
                    self.crossed_objects.add(detection_id)
                    new_counts += 1
                    self.total_count += 1
            else:  # vertical belt, check y-coordinate
                if center[1] > counting_line and detection_id not in self.crossed_objects:
                    self.crossed_objects.add(detection_id)
                    new_counts += 1
                    self.total_count += 1
        
        return new_counts
    
    def process_frame(self, frame):
        """Process a single frame with the selected method"""
        self.frame_number += 1
        
        # Detect chickens using selected method
        if self.method == 'background_subtraction':
            detections, debug_img = self.detect_chickens_background_subtraction(frame)
        elif self.method == 'motion_detection':
            detections, debug_img = self.detect_chickens_motion_detection(frame)
        elif self.method == 'contour_analysis':
            detections, debug_img = self.detect_chickens_contour_analysis(frame)
        elif self.method == 'blob_detection':
            detections, debug_img = self.detect_chickens_blob_detection(frame)
        else:
            detections, debug_img = [], np.zeros_like(frame[:,:,0])
        
        # Track and count
        new_counts = self.track_and_count(detections, frame.shape)
        
        # Draw visualizations
        result_frame = self.draw_results(frame, detections, debug_img)
        
        return result_frame, len(detections), new_counts
    
    def draw_results(self, frame, detections, debug_img):
        """Draw detection results and counting information"""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw counting line
        if self.config.BELT_DIRECTION == 'horizontal':
            line_x = int(w * self.config.COUNTING_LINE_POSITION)
            cv2.line(result, (line_x, 0), (line_x, h), (0, 0, 255), 3)
            cv2.putText(result, 'COUNT LINE', (line_x + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:  # vertical
            line_y = int(h * self.config.COUNTING_LINE_POSITION)
            cv2.line(result, (0, line_y), (w, line_y), (0, 0, 255), 3)
            cv2.putText(result, 'COUNT LINE', (10, line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw detections
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            center = detection['center']
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result, center, 3, (0, 255, 0), -1)
            
            # Draw detection info
            info = f"D{i+1}: A={detection['area']:.0f}"
            cv2.putText(result, info, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw statistics
        cv2.putText(result, f'TOTAL COUNT: {self.total_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(result, f'CURRENT DETECTIONS: {len(detections)}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(result, f'METHOD: {self.method.upper()}', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(result, f'FRAME: {self.frame_number}', (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show debug image if enabled
        if self.config.DEBUG_MODE and debug_img is not None:
            # Resize debug image to fit in corner
            debug_small = cv2.resize(debug_img, (w//4, h//4))
            if len(debug_small.shape) == 2:
                debug_small = cv2.cvtColor(debug_small, cv2.COLOR_GRAY2BGR)
            
            # Place debug image in top-right corner
            result[10:10+debug_small.shape[0], w-debug_small.shape[1]-10:w-10] = debug_small
            cv2.rectangle(result, (w-debug_small.shape[1]-10, 10), 
                         (w-10, 10+debug_small.shape[0]), (255, 255, 255), 2)
            cv2.putText(result, 'DEBUG', (w-debug_small.shape[1]-5, 10+debug_small.shape[0]+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result

def test_multiple_methods(video_source, methods_to_test=None):
    """Test multiple detection methods and compare results"""
    if methods_to_test is None:
        methods_to_test = ['background_subtraction', 'motion_detection', 'contour_analysis', 'blob_detection']
    
    print("Testing multiple detection methods...")
    print("="*60)
    
    for method in methods_to_test:
        print(f"\nTesting {method}...")
        
        # Create config for this method
        config = ChickenCounterConfig()
        config.DETECTION_METHOD = method
        config.SAVE_OUTPUT = False  # Don't save during testing
        config.DEBUG_MODE = True
        
        # Test on first 100 frames
        cap = cv2.VideoCapture(video_source)
        counter = AlternativeChickenCounter(config)
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 100:  # Test first 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            result_frame, detections, new_counts = counter.process_frame(frame)
            
            # Show result for this method
            cv2.imshow(f'{method} - Press any key for next method', result_frame)
            
            # Quick preview - advance on any key
            if cv2.waitKey(1) & 0xFF != 255:
                break
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"  Frames processed: {frame_count}")
        print(f"  Total chickens counted: {counter.total_count}")
        print(f"  Processing FPS: {fps:.2f}")
        print(f"  Final detections in last frame: {detections}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Wait for user input
        input(f"Press Enter to test next method...")

def main_alternative_processing(video_source, method='background_subtraction'):
    """Main processing function for alternative chicken counting"""
    print("="*60)
    print("ALTERNATIVE CHICKEN COUNTER")
    print("="*60)
    
    # Setup configuration
    config = ChickenCounterConfig()
    config.DETECTION_METHOD = method
    
    print(f"Using detection method: {method}")
    print(f"Belt direction: {config.BELT_DIRECTION}")
    print(f"Counting line position: {config.COUNTING_LINE_POSITION*100}%")
    
    # Initialize counter
    counter = AlternativeChickenCounter(config)
    
    # Setup video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_source}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Video writer
    if config.SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))
    else:
        out = None
    
    # Processing loop
    start_time = time.time()
    frame_count = 0
    paused = False
    
    print("\nProcessing started...")
    print("Controls: 'q'=quit, 'space'=pause, 't'=test all methods, 'r'=reset count")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                result_frame, current_detections, new_counts = counter.process_frame(frame)
                
                # Save if enabled
                if out:
                    out.write(result_frame)
                
                # Display
                cv2.imshow('Alternative Chicken Counter', result_frame)
                
                # Progress reporting
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Frame {frame_count}: {counter.total_count} total | "
                          f"Current detections: {current_detections} | FPS: {fps_actual:.1f}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('t'):
                print("Testing all methods...")
                test_multiple_methods(video_source)
            elif key == ord('r'):
                counter.total_count = 0
                counter.crossed_objects.clear()
                print("Count reset to 0")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Final results
        total_time = time.time() - start_time
        print(f"\n" + "="*50)
        print(f"FINAL RESULTS - {method.upper()}")
        print(f"="*50)
        print(f"Total chickens counted: {counter.total_count}")
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {total_time:.2f} seconds")
        if total_time > 0:
            print(f"Average FPS: {frame_count/total_time:.2f}")
        if config.SAVE_OUTPUT:
            print(f"Output saved: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    # Configuration
    VIDEO_SOURCE = 'test-data/REALISTIC-RECORDING.mp4'  # Your conveyor belt video
    
    # Choose detection method:
    # 'background_subtraction' - Best for stationary camera, moving objects
    # 'motion_detection' - Good for detecting movement
    # 'contour_analysis' - Good for shape-based detection
    # 'blob_detection' - Good for round/oval objects
    
    METHOD = 'background_subtraction'  # Start with this one
    
    # Test all methods first to see which works best
    print("Would you like to test all methods first? (y/n)")
    response = input().lower()
    
    if response == 'y':
        test_multiple_methods(VIDEO_SOURCE)
    else:
        main_alternative_processing(VIDEO_SOURCE, METHOD)