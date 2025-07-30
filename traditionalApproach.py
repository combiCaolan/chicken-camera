import cv2
import numpy as np
from collections import defaultdict
import time

class ChickenVideoAnalyzer:
    def __init__(self):
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Initialize tracking
        self.tracker_id = 0
        self.tracks = {}
        self.track_colors = {}
        
        # Parameters (adjustable via trackbars)
        self.params = {
            'yellow_hue_low': 15,
            'yellow_hue_high': 35,
            'saturation_low': 50,
            'value_low': 50,
            'min_contour_area': 500,
            'max_contour_area': 8000,
            'morph_kernel_size': 5,
            'gaussian_blur': 5
        }
        
        self.setup_trackbars()
    
    def setup_trackbars(self):
        """Create trackbars for real-time parameter adjustment"""
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Yellow Hue Low', 'Controls', self.params['yellow_hue_low'], 50, self.update_param)
        cv2.createTrackbar('Yellow Hue High', 'Controls', self.params['yellow_hue_high'], 50, self.update_param)
        cv2.createTrackbar('Saturation Low', 'Controls', self.params['saturation_low'], 255, self.update_param)
        cv2.createTrackbar('Value Low', 'Controls', self.params['value_low'], 255, self.update_param)
        cv2.createTrackbar('Min Area', 'Controls', self.params['min_contour_area'], 5000, self.update_param)
        cv2.createTrackbar('Max Area', 'Controls', self.params['max_contour_area'], 15000, self.update_param)
        cv2.createTrackbar('Blur', 'Controls', self.params['gaussian_blur'], 15, self.update_param)
    
    def update_param(self, val):
        """Update parameters from trackbars"""
        self.params['yellow_hue_low'] = cv2.getTrackbarPos('Yellow Hue Low', 'Controls')
        self.params['yellow_hue_high'] = cv2.getTrackbarPos('Yellow Hue High', 'Controls')
        self.params['saturation_low'] = cv2.getTrackbarPos('Saturation Low', 'Controls')
        self.params['value_low'] = cv2.getTrackbarPos('Value Low', 'Controls')
        self.params['min_contour_area'] = cv2.getTrackbarPos('Min Area', 'Controls')
        self.params['max_contour_area'] = cv2.getTrackbarPos('Max Area', 'Controls')
        self.params['gaussian_blur'] = max(1, cv2.getTrackbarPos('Blur', 'Controls'))
        if self.params['gaussian_blur'] % 2 == 0:
            self.params['gaussian_blur'] += 1
    
    def yellow_color_segmentation(self, frame):
        """Segment yellow chickens using HSV color space"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (self.params['gaussian_blur'], self.params['gaussian_blur']), 0)
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range
        lower_yellow = np.array([self.params['yellow_hue_low'], self.params['saturation_low'], self.params['value_low']])
        upper_yellow = np.array([self.params['yellow_hue_high'], 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up mask
        kernel = np.ones((self.params['morph_kernel_size'], self.params['morph_kernel_size']), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def background_subtraction(self, frame):
        """Extract foreground using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        return fg_mask
    
    def find_contours_and_filter(self, mask):
        """Find and filter contours based on size and shape"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params['min_contour_area'] < area < self.params['max_contour_area']:
                # Additional filtering by aspect ratio and solidity
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if 0.3 < aspect_ratio < 3.0 and solidity > 0.5:
                    valid_contours.append(contour)
        
        return valid_contours
    
    def blob_detection(self, mask):
        """Detect blobs using SimpleBlobDetector"""
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.params['min_contour_area']
        params.maxArea = self.params['max_contour_area']
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.2
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.4
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        
        return keypoints
    
    def simple_tracking(self, contours, frame_idx):
        """Simple centroid-based tracking"""
        current_centroids = []
        
        # Get centroids of current detections
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_centroids.append((cx, cy))
        
        # Simple distance-based tracking
        max_distance = 50
        matched_tracks = {}
        
        for centroid in current_centroids:
            best_match = None
            min_distance = float('inf')
            
            for track_id, track_history in self.tracks.items():
                if len(track_history) > 0:
                    last_pos = track_history[-1]
                    distance = np.sqrt((centroid[0] - last_pos[0])**2 + (centroid[1] - last_pos[1])**2)
                    if distance < min_distance and distance < max_distance:
                        min_distance = distance
                        best_match = track_id
            
            if best_match is not None:
                matched_tracks[best_match] = centroid
            else:
                # New track
                self.tracker_id += 1
                matched_tracks[self.tracker_id] = centroid
                self.track_colors[self.tracker_id] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
        
        # Update tracks
        self.tracks = {track_id: self.tracks.get(track_id, []) + [pos] 
                      for track_id, pos in matched_tracks.items()}
        
        # Remove old tracks
        self.tracks = {k: v[-30:] for k, v in self.tracks.items() if len(v) > 0}
        
        return matched_tracks
    
    def draw_results(self, frame, contours, keypoints, tracks, counts):
        """Draw all detection results on frame"""
        result = frame.copy()
        
        # Draw contours
        if contours:
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Draw blob keypoints
        if keypoints:
            result = cv2.drawKeypoints(result, keypoints, result, (0, 0, 255), 
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Draw tracks
        for track_id, positions in self.tracks.items():
            if len(positions) > 1:
                color = self.track_colors.get(track_id, (255, 255, 255))
                for i in range(1, len(positions)):
                    cv2.line(result, positions[i-1], positions[i], color, 2)
                
                # Draw current position
                if positions:
                    cv2.circle(result, positions[-1], 5, color, -1)
                    cv2.putText(result, str(track_id), 
                              (positions[-1][0] + 10, positions[-1][1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add count information
        y_pos = 30
        cv2.putText(result, f"Contours: {counts['contours']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 25
        cv2.putText(result, f"Blobs: {counts['blobs']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 25
        cv2.putText(result, f"Tracks: {len(self.tracks)}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return result
    
    def process_video(self, video_path):
        """Main video processing function with continuous looping"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video reached - rewind to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                print("Video ended - rewinding to beginning...")
                continue
            
            frame_count += 1
            
            # Method 1: Yellow color segmentation
            yellow_mask = self.yellow_color_segmentation(frame)
            
            # Method 2: Background subtraction
            bg_mask = self.background_subtraction(frame)
            
            # Combine masks (you can experiment with this)
            combined_mask = cv2.bitwise_or(yellow_mask, bg_mask)
            
            # Find contours
            contours = self.find_contours_and_filter(combined_mask)
            
            # Blob detection
            keypoints = self.blob_detection(combined_mask)
            
            # Tracking
            tracks = self.simple_tracking(contours, frame_count)
            
            # Count detections
            counts = {
                'contours': len(contours),
                'blobs': len(keypoints),
                'tracks': len(self.tracks)
            }
            
            # Draw results
            result_frame = self.draw_results(frame, contours, keypoints, tracks, counts)
            
            # Show different views
            cv2.imshow('Original', frame)
            cv2.imshow('Yellow Mask', yellow_mask)
            cv2.imshow('Background Mask', bg_mask)
            cv2.imshow('Combined Mask', combined_mask)
            cv2.imshow('Detection Results', result_frame)
            
            # Control playback speed
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to pause
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    analyzer = ChickenVideoAnalyzer()
    
    # Replace with your video path
    video_path = "test-data/REALISTIC-RECORDING.mp4"
    
    print("Controls:")
    print("- Use trackbars in 'Controls' window to adjust parameters")
    print("- Press SPACE to pause/resume")
    print("- Press 'q' to quit")
    print("- Experiment with different parameter combinations!")
    
    analyzer.process_video(video_path)

if __name__ == "__main__":
    main()