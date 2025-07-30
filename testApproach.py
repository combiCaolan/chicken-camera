import cv2
import numpy as np

class InteractiveChickenTuner:
    def __init__(self, image_path):
        # Load the image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize if too large for display
        height, width = self.original_image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # Create windows
        cv2.namedWindow('Original Image', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Processed Result', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
        
        # Initialize parameters
        self.params = {
            # Method selection
            'method': 0,  # 0=Simple Threshold, 1=Adaptive, 2=HSV, 3=Morphological, 4=Edge, 5=Combined
            
            # Simple threshold parameters
            'simple_threshold': 127,
            'threshold_type': 0,  # 0=BINARY, 1=BINARY_INV
            
            # Adaptive threshold parameters  
            'adaptive_max_value': 255,
            'adaptive_method': 0,  # 0=MEAN, 1=GAUSSIAN
            'adaptive_block_size': 11,
            'adaptive_c': 2,
            
            # HSV parameters
            'hue_low': 0,
            'hue_high': 180,
            'sat_low': 0,
            'sat_high': 255,
            'val_low': 200,
            'val_high': 255,
            
            # Morphological parameters
            'morph_operation': 0,  # 0=CLOSE, 1=OPEN, 2=GRADIENT, 3=TOPHAT, 4=BLACKHAT
            'morph_kernel_size': 5,
            'morph_iterations': 1,
            
            # Edge detection parameters
            'canny_low': 50,
            'canny_high': 150,
            'canny_aperture': 3,
            
            # Preprocessing parameters
            'gaussian_blur': 1,
            'contrast_alpha': 10,  # Will be divided by 10 (0.1 to 3.0)
            'brightness_beta': 0,
            
            # Post-processing
            'contour_min_area': 100,
            'contour_max_area': 5000,
            'show_contours': 1,
            'show_bounding_boxes': 1
        }
        
        self.setup_trackbars()
        self.update_display()
    
    def setup_trackbars(self):
        """Create all the control trackbars"""
        # Method selection
        cv2.createTrackbar('Method', 'Controls', self.params['method'], 5, self.on_trackbar_change)
        cv2.setTrackbarPos('Method', 'Controls', 0)
        
        # Simple threshold
        cv2.createTrackbar('Simple Threshold', 'Controls', self.params['simple_threshold'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Threshold Type', 'Controls', self.params['threshold_type'], 1, self.on_trackbar_change)
        
        # Adaptive threshold  
        cv2.createTrackbar('Adaptive Max', 'Controls', self.params['adaptive_max_value'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Adaptive Method', 'Controls', self.params['adaptive_method'], 1, self.on_trackbar_change)
        cv2.createTrackbar('Block Size', 'Controls', self.params['adaptive_block_size'], 51, self.on_trackbar_change)
        cv2.createTrackbar('Adaptive C', 'Controls', self.params['adaptive_c'], 20, self.on_trackbar_change)
        
        # HSV parameters
        cv2.createTrackbar('Hue Low', 'Controls', self.params['hue_low'], 180, self.on_trackbar_change)
        cv2.createTrackbar('Hue High', 'Controls', self.params['hue_high'], 180, self.on_trackbar_change)
        cv2.createTrackbar('Sat Low', 'Controls', self.params['sat_low'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Sat High', 'Controls', self.params['sat_high'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Val Low', 'Controls', self.params['val_low'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Val High', 'Controls', self.params['val_high'], 255, self.on_trackbar_change)
        
        # Morphological parameters
        cv2.createTrackbar('Morph Operation', 'Controls', self.params['morph_operation'], 4, self.on_trackbar_change)
        cv2.createTrackbar('Morph Kernel', 'Controls', self.params['morph_kernel_size'], 21, self.on_trackbar_change)
        cv2.createTrackbar('Morph Iterations', 'Controls', self.params['morph_iterations'], 10, self.on_trackbar_change)
        
        # Edge detection
        cv2.createTrackbar('Canny Low', 'Controls', self.params['canny_low'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Canny High', 'Controls', self.params['canny_high'], 255, self.on_trackbar_change)
        cv2.createTrackbar('Canny Aperture', 'Controls', self.params['canny_aperture'], 7, self.on_trackbar_change)
        
        # Preprocessing
        cv2.createTrackbar('Gaussian Blur', 'Controls', self.params['gaussian_blur'], 15, self.on_trackbar_change)
        cv2.createTrackbar('Contrast x10', 'Controls', self.params['contrast_alpha'], 30, self.on_trackbar_change)
        cv2.createTrackbar('Brightness', 'Controls', self.params['brightness_beta'], 100, self.on_trackbar_change)
        
        # Post-processing
        cv2.createTrackbar('Min Area', 'Controls', self.params['contour_min_area'], 2000, self.on_trackbar_change)
        cv2.createTrackbar('Max Area', 'Controls', self.params['contour_max_area'], 10000, self.on_trackbar_change)
        cv2.createTrackbar('Show Contours', 'Controls', self.params['show_contours'], 1, self.on_trackbar_change)
        cv2.createTrackbar('Show Boxes', 'Controls', self.params['show_bounding_boxes'], 1, self.on_trackbar_change)
    
    def on_trackbar_change(self, val):
        """Update parameters when trackbar changes"""
        self.params['method'] = cv2.getTrackbarPos('Method', 'Controls')
        self.params['simple_threshold'] = cv2.getTrackbarPos('Simple Threshold', 'Controls')
        self.params['threshold_type'] = cv2.getTrackbarPos('Threshold Type', 'Controls')
        
        self.params['adaptive_max_value'] = cv2.getTrackbarPos('Adaptive Max', 'Controls')
        self.params['adaptive_method'] = cv2.getTrackbarPos('Adaptive Method', 'Controls')
        self.params['adaptive_block_size'] = max(3, cv2.getTrackbarPos('Block Size', 'Controls'))
        if self.params['adaptive_block_size'] % 2 == 0:
            self.params['adaptive_block_size'] += 1
        self.params['adaptive_c'] = cv2.getTrackbarPos('Adaptive C', 'Controls')
        
        self.params['hue_low'] = cv2.getTrackbarPos('Hue Low', 'Controls')
        self.params['hue_high'] = cv2.getTrackbarPos('Hue High', 'Controls')
        self.params['sat_low'] = cv2.getTrackbarPos('Sat Low', 'Controls')
        self.params['sat_high'] = cv2.getTrackbarPos('Sat High', 'Controls')
        self.params['val_low'] = cv2.getTrackbarPos('Val Low', 'Controls')
        self.params['val_high'] = cv2.getTrackbarPos('Val High', 'Controls')
        
        self.params['morph_operation'] = cv2.getTrackbarPos('Morph Operation', 'Controls')
        self.params['morph_kernel_size'] = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls'))
        if self.params['morph_kernel_size'] % 2 == 0:
            self.params['morph_kernel_size'] += 1
        self.params['morph_iterations'] = max(1, cv2.getTrackbarPos('Morph Iterations', 'Controls'))
        
        self.params['canny_low'] = cv2.getTrackbarPos('Canny Low', 'Controls')
        self.params['canny_high'] = cv2.getTrackbarPos('Canny High', 'Controls')
        aperture = cv2.getTrackbarPos('Canny Aperture', 'Controls')
        self.params['canny_aperture'] = 3 if aperture < 3 else (5 if aperture < 5 else 7)
        
        self.params['gaussian_blur'] = max(1, cv2.getTrackbarPos('Gaussian Blur', 'Controls'))
        if self.params['gaussian_blur'] % 2 == 0:
            self.params['gaussian_blur'] += 1
        self.params['contrast_alpha'] = cv2.getTrackbarPos('Contrast x10', 'Controls') / 10.0
        self.params['brightness_beta'] = cv2.getTrackbarPos('Brightness', 'Controls') - 50
        
        self.params['contour_min_area'] = cv2.getTrackbarPos('Min Area', 'Controls')
        self.params['contour_max_area'] = cv2.getTrackbarPos('Max Area', 'Controls')
        self.params['show_contours'] = cv2.getTrackbarPos('Show Contours', 'Controls')
        self.params['show_bounding_boxes'] = cv2.getTrackbarPos('Show Boxes', 'Controls')
        
        self.update_display()
    
    def preprocess_image(self, image):
        """Apply preprocessing based on current parameters"""
        # Apply contrast and brightness
        processed = cv2.convertScaleAbs(image, alpha=self.params['contrast_alpha'], beta=self.params['brightness_beta'])
        
        # Apply Gaussian blur
        if self.params['gaussian_blur'] > 1:
            processed = cv2.GaussianBlur(processed, (self.params['gaussian_blur'], self.params['gaussian_blur']), 0)
        
        return processed
    
    def apply_method(self, processed_image):
        """Apply the selected computer vision method"""
        method = self.params['method']
        
        if method == 0:  # Simple Threshold
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            threshold_type = cv2.THRESH_BINARY if self.params['threshold_type'] == 0 else cv2.THRESH_BINARY_INV
            _, binary = cv2.threshold(gray, self.params['simple_threshold'], 255, threshold_type)
            return binary
            
        elif method == 1:  # Adaptive Threshold
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if self.params['adaptive_method'] == 0 else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            binary = cv2.adaptiveThreshold(gray, self.params['adaptive_max_value'], adaptive_method, 
                                         cv2.THRESH_BINARY, self.params['adaptive_block_size'], self.params['adaptive_c'])
            return binary
            
        elif method == 2:  # HSV Color Segmentation
            hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
            lower = np.array([self.params['hue_low'], self.params['sat_low'], self.params['val_low']])
            upper = np.array([self.params['hue_high'], self.params['sat_high'], self.params['val_high']])
            mask = cv2.inRange(hsv, lower, upper)
            return mask
            
        elif method == 3:  # Morphological Operations
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, self.params['simple_threshold'], 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((self.params['morph_kernel_size'], self.params['morph_kernel_size']), np.uint8)
            morph_ops = [cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]
            operation = morph_ops[self.params['morph_operation']]
            
            result = cv2.morphologyEx(binary, operation, kernel, iterations=self.params['morph_iterations'])
            return result
            
        elif method == 4:  # Edge Detection
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.params['canny_low'], self.params['canny_high'], 
                            apertureSize=self.params['canny_aperture'])
            return edges
            
        elif method == 5:  # Combined Method
            # Combine threshold and morphological operations
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, self.params['simple_threshold'], 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((self.params['morph_kernel_size'], self.params['morph_kernel_size']), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.params['morph_iterations'])
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return binary
    
    def find_and_draw_contours(self, binary_image, original_image):
        """Find contours and draw them on the original image"""
        result = original_image.copy()
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.params['contour_min_area'] <= area <= self.params['contour_max_area']:
                valid_contours.append(contour)
        
        # Draw contours
        if self.params['show_contours'] and valid_contours:
            cv2.drawContours(result, valid_contours, -1, (0, 255, 0), 2)
        
        # Draw bounding boxes
        if self.params['show_bounding_boxes'] and valid_contours:
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Add count
        count = len(valid_contours)
        cv2.putText(result, f'Chickens: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add method name
        method_names = ['Simple Threshold', 'Adaptive Threshold', 'HSV Segmentation', 
                       'Morphological Ops', 'Edge Detection', 'Combined Method']
        cv2.putText(result, method_names[self.params['method']], (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return result, count
    
    def update_display(self):
        """Update the display with current parameters"""
        try:
            # Preprocess image
            processed = self.preprocess_image(self.original_image)
            
            # Apply selected method
            binary_result = self.apply_method(processed)
            
            # Find contours and create result image
            result_image, count = self.find_and_draw_contours(binary_result, processed)
            
            # Convert binary to 3-channel for display
            if len(binary_result.shape) == 2:
                binary_display = cv2.cvtColor(binary_result, cv2.COLOR_GRAY2BGR)
            else:
                binary_display = binary_result
            
            # Create side-by-side display
            combined_display = np.hstack([result_image, binary_display])
            
            # Display images
            cv2.imshow('Original Image', self.original_image)
            cv2.imshow('Processed Result', combined_display)
            
        except Exception as e:
            print(f"Error in processing: {e}")
    
    def run(self):
        """Main loop"""
        print("Interactive Chicken Detection Tuner")
        print("=" * 40)
        print("Methods:")
        print("0 = Simple Threshold")
        print("1 = Adaptive Threshold") 
        print("2 = HSV Color Segmentation")
        print("3 = Morphological Operations")
        print("4 = Edge Detection")
        print("5 = Combined Method")
        print("\nControls:")
        print("- Adjust sliders to tune parameters")
        print("- Press 's' to save current parameters")
        print("- Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\nCurrent Parameters:")
                for param, value in self.params.items():
                    print(f"{param}: {value}")
        
        cv2.destroyAllWindows()

def main():
    # Replace with your image path
    image_path = "test-data/gimagesphoto.jpg"
    
    try:
        tuner = InteractiveChickenTuner(image_path)
        tuner.run()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure the image path is correct!")

if __name__ == "__main__":
    main()