import cv2
import numpy as np

class OptimizedChickenCounter:
    def __init__(self):
        # Background subtractor - key for moving conveyor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,  # Disable shadows for better performance
            varThreshold=30,      # Lower threshold for subtle movements
            history=200           # Learn background faster
        )
        
        # Morphological kernels for cleaning blobs
        self.cleanup_kernel = np.ones((7, 7), np.uint8)
        self.separate_kernel = np.ones((3, 3), np.uint8)
        
        # Chicken size parameters (adjust based on your video)
        self.min_chicken_area = 300   # Minimum pixels for a chicken
        self.max_chicken_area = 4000  # Maximum pixels for a chicken
        
        # Blob detector setup
        self.setup_blob_detector()
    
    def setup_blob_detector(self):
        """Configure blob detector specifically for blurry chickens"""
        params = cv2.SimpleBlobDetector_Params()
        
        # Area filtering - critical for chicken counting
        params.filterByArea = True
        params.minArea = self.min_chicken_area
        params.maxArea = self.max_chicken_area
        
        # Relaxed circularity for blurry shapes
        params.filterByCircularity = True
        params.minCircularity = 0.2  # Very permissive for blur
        
        # Convexity helps distinguish chickens from noise
        params.filterByConvexity = True
        params.minConvexity = 0.3
        
        # Disable inertia (not useful for blurry objects)
        params.filterByInertia = False
        
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
    
    def preprocess_frame(self, frame):
        """Preprocess frame to improve detection"""
        # Gaussian blur to reduce noise while preserving blob shapes
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Contrast enhancement for better separation
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
        
        return blurred, enhanced
    
    def background_subtraction_method(self, frame):
        """Extract moving objects using background subtraction"""
        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask with morphological operations
        # Close gaps in chicken blobs
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.cleanup_kernel, iterations=2)
        # Remove small noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.separate_kernel, iterations=1)
        
        return cleaned
    
    def threshold_method(self, enhanced_gray):
        """Simple thresholding as backup method"""
        # Adaptive threshold works better with uneven lighting
        thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 15, -10)
        
        # Invert so chickens are white (255)
        thresh = cv2.bitwise_not(thresh)
        
        # Clean up
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.cleanup_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.separate_kernel)
        
        return cleaned
    
    def count_with_contours(self, mask):
        """Count chickens using contour analysis"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_chickens = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_chicken_area < area < self.max_chicken_area:
                # Additional shape filtering
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Very relaxed circularity for blurry shapes
                    if circularity > 0.1:
                        valid_chickens.append(contour)
        
        return len(valid_chickens), valid_chickens
    
    def count_with_blobs(self, mask):
        """Count chickens using blob detection"""
        keypoints = self.blob_detector.detect(mask)
        return len(keypoints), keypoints
    
    def process_frame(self, frame):
        """Main processing function combining all methods"""
        # Preprocess
        blurred_frame, enhanced_gray = self.preprocess_frame(frame)
        
        # Method 1: Background subtraction (primary)
        bg_mask = self.background_subtraction_method(frame)
        
        # Method 2: Thresholding (backup)
        thresh_mask = self.threshold_method(enhanced_gray)
        
        # Combine masks (background subtraction weighted more heavily)
        combined_mask = cv2.addWeighted(bg_mask, 0.7, thresh_mask, 0.3, 0)
        combined_mask = (combined_mask > 127).astype(np.uint8) * 255
        
        # Count using both methods
        contour_count, contours = self.count_with_contours(combined_mask)
        blob_count, keypoints = self.count_with_blobs(combined_mask)
        
        # Use the more conservative count (usually more accurate)
        final_count = min(contour_count, blob_count) if blob_count > 0 else contour_count
        
        return {
            'count': final_count,
            'contour_count': contour_count,
            'blob_count': blob_count,
            'combined_mask': combined_mask,
            'bg_mask': bg_mask,
            'thresh_mask': thresh_mask,
            'contours': contours,
            'keypoints': keypoints
        }
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        output = frame.copy()
        
        # Draw contours in green
        if results['contours']:
            cv2.drawContours(output, results['contours'], -1, (0, 255, 0), 2)
            # Add bounding boxes
            for contour in results['contours']:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Draw blob keypoints in red
        if results['keypoints']:
            output = cv2.drawKeypoints(output, results['keypoints'], output, 
                                     (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Add count information
        cv2.putText(output, f"Chickens: {results['count']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"Contours: {results['contour_count']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"Blobs: {results['blob_count']}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return output
    
    def process_video(self, video_path):
        """Process video with continuous looping"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        print("Controls:")
        print("- Press SPACE to pause/resume")
        print("- Press 'q' to quit") 
        print("- Press 'r' to reset background model")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Rewind video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Video rewound - continuing...")
                continue
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw results
            output_frame = self.draw_results(frame, results)
            
            # Display windows
            cv2.imshow('Original', frame)
            cv2.imshow('Detection Results', output_frame)
            cv2.imshow('Combined Mask', results['combined_mask'])
            cv2.imshow('Background Subtraction', results['bg_mask'])
            
            # Handle controls
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause
                cv2.waitKey(0)
            elif key == ord('r'):  # Reset background
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    detectShadows=False, varThreshold=30, history=200)
                print("Background model reset")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize the counter
    counter = OptimizedChickenCounter()
    
    # Replace with your video path
    video_path = "test-data/REALISTIC-RECORDING.mp4"
    
    # Process the video
    counter.process_video(video_path)

if __name__ == "__main__":
    main()