import cv2
import numpy as np
import json
from datetime import datetime

class InteractiveChickenTuner:
    """Real-time parameter tuning interface for chicken detection"""
    
    def __init__(self, image_path):
        # Load original image
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create copies for processing
        self.result_img = self.original_img.copy()
        self.gray = None
        self.thresh = None
        
        # Parameter values (will be controlled by trackbars)
        self.params = {
            # Image adjustment
            'brightness': 100,     # 0-200 (100 = normal)
            
            # Gaussian blur parameters
            'blur_kernel': 3,      # 1-15 (odd numbers only)
            'blur_sigma': 2,       # 0-10
            
            # Adaptive threshold parameters
            'thresh_max': 255,     # 100-300
            'thresh_block': 85,    # 3-201 (odd numbers only)
            'thresh_c': 21,        # 0-50 (will be negated)
            
            # Morphological operations
            'morph_kernel': 9,     # 1-21 (odd numbers only)
            'erode_iter': 2,       # 0-10
            'dilate_iter': 1,      # 0-10
            'final_blur_kernel': 3, # 1-15 (odd numbers only)
            'final_blur_sigma': 1,  # 0-10
            
            # Contour filtering
            'min_area': 500,       # 0-2000
            'max_area': 10000,     # 1000-50000
            'min_aspect': 30,      # 10-100 (will be divided by 100)
            'max_aspect': 300,     # 100-500 (will be divided by 100)
        }
        
        # ROI (Region of Interest) selection - brush-based
        self.roi_mask = None  # Will store brush-painted mask
        self.use_roi = False
        self.roi_selected = False
        self.brush_size = 50  # Default brush size
        
        # Button states for trackbar-based buttons
        self.button_states = {
            'reset_button': 0,
            'save_button': 0
        }
        
        # Statistics initialization
        self.stats = {
            'total_contours': 0,
            'filtered_contours': 0,
            'detection_ratio': 0.0
        }
        
        # Initialization flag
        self.initialization_complete = False
        
        # Start with ROI selection
        if not self.select_roi():
            print("ROI selection cancelled")
            raise ValueError("ROI selection cancelled")
        
        # Create windows and trackbars only after ROI is selected
        self.setup_windows()
        self.create_trackbars()
        
        # Mark initialization as complete
        self.initialization_complete = True
        
        # Initial processing
        self.process_image()
    
    def select_roi(self):
        """Interactive ROI selection with brush painting"""
        print("\n🎨 REGION OF INTEREST BRUSH SELECTION")
        print("="*45)
        print("Instructions:")
        print("1. Click and drag to PAINT the areas for chicken detection")
        print("2. Use mouse wheel or +/- keys to adjust brush size")
        print("3. Press SPACE to confirm painted area") 
        print("4. Press 'f' to use full image (no ROI)")
        print("5. Press 'c' to clear all painting")
        print("6. Press 'u' to undo last stroke")
        print("7. Press 'q' to cancel")
        
        # Create ROI selection window
        cv2.namedWindow('ROI Brush Selection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ROI Brush Selection', 1000, 700)
        
        # Initialize brush mask (same size as original image)
        h, w = self.original_img.shape[:2]
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Brush painting variables
        painting = False
        last_point = None
        brush_strokes = []  # For undo functionality
        current_stroke = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal painting, last_point, current_stroke
            
            if event == cv2.EVENT_LBUTTONDOWN:
                painting = True
                last_point = (x, y)
                current_stroke = [(x, y)]
                # Paint initial point
                cv2.circle(self.roi_mask, (x, y), self.brush_size, 255, -1)
                
            elif event == cv2.EVENT_MOUSEMOVE and painting:
                if last_point:
                    # Draw line from last point to current point for smooth brush
                    cv2.line(self.roi_mask, last_point, (x, y), 255, self.brush_size * 2)
                    cv2.circle(self.roi_mask, (x, y), self.brush_size, 255, -1)
                    last_point = (x, y)
                    current_stroke.append((x, y))
            
            elif event == cv2.EVENT_LBUTTONUP:
                painting = False
                if current_stroke:
                    brush_strokes.append(current_stroke.copy())
                    current_stroke = []
                last_point = None
            
            elif event == cv2.EVENT_MOUSEWHEEL:
                # Adjust brush size with mouse wheel
                if flags > 0:  # Scroll up
                    self.brush_size = min(100, self.brush_size + 5)
                else:  # Scroll down
                    self.brush_size = max(5, self.brush_size - 5)
                print(f"Brush size: {self.brush_size}")
        
        # Set mouse callback
        cv2.setMouseCallback('ROI Brush Selection', mouse_callback)
        
        # Main brush selection loop
        while True:
            # Create display image
            display_img = self.original_img.copy()
            
            # Overlay the painted mask in semi-transparent green
            if np.any(self.roi_mask > 0):
                colored_mask = np.zeros_like(display_img)
                colored_mask[self.roi_mask > 0] = [0, 255, 0]  # Green overlay
                display_img = cv2.addWeighted(display_img, 0.7, colored_mask, 0.3, 0)
            
            # Draw current brush cursor if mouse is over window
            try:
                # Get mouse position (this is a bit tricky in OpenCV)
                mouse_pos = cv2.getWindowProperty('ROI Brush Selection', cv2.WND_PROP_AUTOSIZE)
            except:
                mouse_pos = None
            
            # Add instructions overlay
            cv2.putText(display_img, 'PAINT areas for chicken detection', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_img, f'Brush size: {self.brush_size} (mouse wheel or +/- to adjust)', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Calculate painted area statistics
            painted_pixels = np.sum(self.roi_mask > 0)
            total_pixels = h * w
            coverage_percent = (painted_pixels / total_pixels) * 100
            
            cv2.putText(display_img, f'Painted area: {coverage_percent:.1f}% of image', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Control instructions
            cv2.putText(display_img, 'SPACE=confirm, F=full image, C=clear, U=undo, Q=cancel', 
                       (10, display_img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('ROI Brush Selection', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - confirm selection
                if np.any(self.roi_mask > 0):
                    self.use_roi = True
                    self.roi_selected = True
                    painted_area = np.sum(self.roi_mask > 0)
                    print(f"✅ ROI confirmed: {painted_area} pixels painted ({coverage_percent:.1f}% of image)")
                    cv2.destroyWindow('ROI Brush Selection')
                    return True
                else:
                    print("❌ No area painted. Please paint some areas first.")
            
            elif key == ord('f'):  # F - use full image
                self.use_roi = False
                self.roi_selected = True
                self.roi_mask = None
                print("✅ Using full image (no ROI)")
                cv2.destroyWindow('ROI Brush Selection')
                return True
            
            elif key == ord('c'):  # C - clear all painting
                self.roi_mask = np.zeros((h, w), dtype=np.uint8)
                brush_strokes.clear()
                current_stroke.clear()
                print("🎨 Cleared all painting")
            
            elif key == ord('u'):  # U - undo last stroke
                if brush_strokes:
                    # Remove last stroke
                    brush_strokes.pop()
                    # Recreate mask from remaining strokes
                    self.roi_mask = np.zeros((h, w), dtype=np.uint8)
                    for stroke in brush_strokes:
                        for i, point in enumerate(stroke):
                            cv2.circle(self.roi_mask, point, self.brush_size, 255, -1)
                            if i > 0:
                                cv2.line(self.roi_mask, stroke[i-1], point, 255, self.brush_size * 2)
                    print("↶ Undid last brush stroke")
                else:
                    print("❌ Nothing to undo")
            
            elif key == ord('+') or key == ord('='):  # + key - increase brush size
                self.brush_size = min(100, self.brush_size + 5)
                print(f"🖌️ Brush size: {self.brush_size}")
            
            elif key == ord('-') or key == ord('_'):  # - key - decrease brush size
                self.brush_size = max(5, self.brush_size - 5)
                print(f"🖌️ Brush size: {self.brush_size}")
            
            elif key == ord('q'):  # Q - cancel
                print("❌ ROI brush selection cancelled")
                cv2.destroyWindow('ROI Brush Selection')
                return False
        
        # Statistics
        self.stats = {
            'total_contours': 0,
            'filtered_contours': 0,
            'detection_ratio': 0.0
        }
        
        # Create windows
        self.setup_windows()
        
        # Create trackbars
        self.create_trackbars()
        
        # Initial processing
        self.process_image()
    
    def setup_windows(self):
        """Create and position windows"""
        # Main result window
        cv2.namedWindow('Chicken Detection - Real-time Tuning', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chicken Detection - Real-time Tuning', 800, 600)
        cv2.moveWindow('Chicken Detection - Real-time Tuning', 100, 100)
        
        # Processing steps windows
        cv2.namedWindow('1. Grayscale + Blur', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('1. Grayscale + Blur', 400, 300)
        cv2.moveWindow('1. Grayscale + Blur', 950, 100)
        
        cv2.namedWindow('2. Threshold', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('2. Threshold', 400, 300)
        cv2.moveWindow('2. Threshold', 950, 450)
        
        cv2.namedWindow('3. Morphological', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('3. Morphological', 400, 300)
        cv2.moveWindow('3. Morphological', 1400, 100)
        
        # Control panel window (for trackbars)
        cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Control Panel', 400, 800)  # Made taller for new controls
        cv2.moveWindow('Control Panel', 1400, 450)
    
    def create_trackbars(self):
        """Create all parameter control trackbars"""
        window = 'Control Panel'
        
        # Image Adjustment Controls
        cv2.createTrackbar('Brightness', window, self.params['brightness'], 200, self.on_trackbar_change)
        
        # Brush ROI Controls (only show if ROI is being used)
        if self.use_roi:
            cv2.createTrackbar('Brush Size', window, self.brush_size, 100, self.on_brush_size_change)
        
        # Action Buttons (implemented as trackbars)
        cv2.createTrackbar('RESET (move to reset)', window, 0, 1, self.on_reset_button)
        cv2.createTrackbar('SAVE (move to save)', window, 0, 1, self.on_save_button)
        cv2.createTrackbar('ROI (move to reselect)', window, 0, 1, self.on_roi_button)
        
        # Gaussian Blur Controls
        cv2.createTrackbar('Blur Kernel', window, self.params['blur_kernel'], 15, self.on_trackbar_change)
        cv2.createTrackbar('Blur Sigma', window, self.params['blur_sigma'], 10, self.on_trackbar_change)
        
        # Adaptive Threshold Controls
        cv2.createTrackbar('Threshold Max', window, self.params['thresh_max'], 300, self.on_trackbar_change)
        cv2.createTrackbar('Threshold Block', window, self.params['thresh_block'], 201, self.on_trackbar_change)
        cv2.createTrackbar('Threshold C', window, self.params['thresh_c'], 50, self.on_trackbar_change)
        
        # Morphological Controls
        cv2.createTrackbar('Morph Kernel', window, self.params['morph_kernel'], 21, self.on_trackbar_change)
        cv2.createTrackbar('Erode Iterations', window, self.params['erode_iter'], 10, self.on_trackbar_change)
        cv2.createTrackbar('Dilate Iterations', window, self.params['dilate_iter'], 10, self.on_trackbar_change)
        cv2.createTrackbar('Final Blur Kernel', window, self.params['final_blur_kernel'], 15, self.on_trackbar_change)
        cv2.createTrackbar('Final Blur Sigma', window, self.params['final_blur_sigma'], 10, self.on_trackbar_change)
        
        # Contour Filtering Controls
        cv2.createTrackbar('Min Area', window, self.params['min_area'], 2000, self.on_trackbar_change)
        cv2.createTrackbar('Max Area', window, self.params['max_area'], 50000, self.on_trackbar_change)
        cv2.createTrackbar('Min Aspect x100', window, self.params['min_aspect'], 100, self.on_trackbar_change)
        cv2.createTrackbar('Max Aspect x100', window, self.params['max_aspect'], 500, self.on_trackbar_change)
        
        # Create empty image for control panel with instructions
        control_img = np.zeros((800, 400, 3), dtype=np.uint8)  # Made taller for new controls
        
        # Add instructions
        instructions = [
            "REAL-TIME CHICKEN DETECTION TUNER",
            "",
            "BUTTON CONTROLS:",
            "- RESET slider: Move to reset all params",
            "- SAVE slider: Move to save current params",
            "- ROI slider: Move to reselect detection area",
            "- Brightness: Adjust image brightness",
            "",
            "Adjust parameters using sliders above",
            "Watch results update in real-time",
            "",
            "KEYBOARD CONTROLS:",
            "- 'r': Reset to defaults",
            "- 's': Save current parameters",
            "- 'l': Load saved parameters", 
            "- 'o': Reselect ROI area",
            "- 'p': Print current parameters",
            "- 'q': Quit",
            "",
            "TIP: Start with ROI selection,",
            "then brightness, then threshold,",
            "finally tune morphology & filtering"
        ]
        
        y_offset = 30
        for instruction in instructions:
            color = (255, 255, 255) if instruction != "" else (100, 100, 100)
            if instruction.startswith("REAL-TIME"):
                color = (0, 255, 255)
            elif instruction.startswith(("BUTTON", "KEYBOARD")):
                color = (0, 255, 0)
            elif instruction.startswith("TIP"):
                color = (0, 165, 255)
            
            cv2.putText(control_img, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        cv2.imshow(window, control_img)
    
    def on_trackbar_change(self, val):
        """Callback function for trackbar changes - triggers reprocessing"""
        if not hasattr(self, 'initialization_complete') or not self.initialization_complete:
            return  # Skip if not fully initialized
        
        try:
            self.update_parameters()
            self.process_image()
        except Exception as e:
            print(f"Error in trackbar callback: {e}")
            pass
    
    def on_reset_button(self, val):
        """Callback for reset button trackbar"""
        if val == 1:  # Button was moved to 1
            try:
                print("🔄 RESET button pressed - resetting to defaults...")
                self.reset_to_defaults()
                # Reset button trackbar back to 0
                cv2.setTrackbarPos('RESET (move to reset)', 'Control Panel', 0)
            except Exception as e:
                print(f"Error in reset callback: {e}")
    
    def on_save_button(self, val):
        """Callback for save button trackbar"""
        if val == 1:  # Button was moved to 1
            try:
                print("💾 SAVE button pressed - saving parameters...")
                filename = self.save_parameters()
                print(f"✅ Saved to: {filename}")
                # Reset button trackbar back to 0
                cv2.setTrackbarPos('SAVE (move to save)', 'Control Panel', 0)
            except Exception as e:
                print(f"Error in save callback: {e}")
    
    def on_roi_button(self, val):
        """Callback for ROI reselection button trackbar"""
        if val == 1:  # Button was moved to 1
            try:
                print("🎨 ROI button pressed - reselecting area...")
                # Hide current windows temporarily
                cv2.destroyAllWindows()
                # Reselect ROI
                if self.select_roi():
                    # Recreate windows and reprocess
                    self.setup_windows()
                    self.create_trackbars()
                    # Restore all trackbar positions
                    self.restore_trackbar_positions()
                    self.process_image()
                else:
                    print("ROI reselection cancelled")
                # Reset button trackbar back to 0 (will be recreated)
                try:
                    cv2.setTrackbarPos('ROI (move to reselect)', 'Control Panel', 0)
                except:
                    pass  # Window might not exist yet
            except Exception as e:
                print(f"Error in ROI callback: {e}")
    
    def on_brush_size_change(self, val):
        """Callback for brush size trackbar"""
        try:
            self.brush_size = max(5, val)  # Minimum brush size of 5
            # No need to reprocess image, this only affects future brush strokes
        except Exception as e:
            print(f"Error in brush size callback: {e}")
    
    def restore_trackbar_positions(self):
        """Restore all trackbar positions after recreating windows"""
        window = 'Control Panel'
        try:
            cv2.setTrackbarPos('Brightness', window, self.params['brightness'])
            if self.use_roi:
                cv2.setTrackbarPos('Brush Size', window, self.brush_size)
            cv2.setTrackbarPos('Blur Kernel', window, self.params['blur_kernel'])
            cv2.setTrackbarPos('Blur Sigma', window, self.params['blur_sigma'])
            cv2.setTrackbarPos('Threshold Max', window, self.params['thresh_max'])
            cv2.setTrackbarPos('Threshold Block', window, self.params['thresh_block'])
            cv2.setTrackbarPos('Threshold C', window, self.params['thresh_c'])
            cv2.setTrackbarPos('Morph Kernel', window, self.params['morph_kernel'])
            cv2.setTrackbarPos('Erode Iterations', window, self.params['erode_iter'])
            cv2.setTrackbarPos('Dilate Iterations', window, self.params['dilate_iter'])
            cv2.setTrackbarPos('Final Blur Kernel', window, self.params['final_blur_kernel'])
            cv2.setTrackbarPos('Final Blur Sigma', window, self.params['final_blur_sigma'])
            cv2.setTrackbarPos('Min Area', window, self.params['min_area'])
            cv2.setTrackbarPos('Max Area', window, self.params['max_area'])
            cv2.setTrackbarPos('Min Aspect x100', window, self.params['min_aspect'])
            cv2.setTrackbarPos('Max Aspect x100', window, self.params['max_aspect'])
            print("✓ Trackbar positions restored")
        except Exception as e:
            print(f"Note: Could not restore all trackbar positions: {e}")
            pass  # Some trackbars might not exist yet
    
    def update_parameters(self):
        """Update parameters from trackbar values"""
        window = 'Control Panel'
        
        # Check if window exists first
        try:
            # Get all trackbar values with error handling
            self.params['brightness'] = cv2.getTrackbarPos('Brightness', window)
            
            self.params['blur_kernel'] = max(1, cv2.getTrackbarPos('Blur Kernel', window))
            self.params['blur_sigma'] = cv2.getTrackbarPos('Blur Sigma', window)
            
            self.params['thresh_max'] = max(100, cv2.getTrackbarPos('Threshold Max', window))
            self.params['thresh_block'] = max(3, cv2.getTrackbarPos('Threshold Block', window))
            self.params['thresh_c'] = cv2.getTrackbarPos('Threshold C', window)
            
            self.params['morph_kernel'] = max(1, cv2.getTrackbarPos('Morph Kernel', window))
            self.params['erode_iter'] = cv2.getTrackbarPos('Erode Iterations', window)
            self.params['dilate_iter'] = cv2.getTrackbarPos('Dilate Iterations', window)
            self.params['final_blur_kernel'] = max(1, cv2.getTrackbarPos('Final Blur Kernel', window))
            self.params['final_blur_sigma'] = cv2.getTrackbarPos('Final Blur Sigma', window)
            
            self.params['min_area'] = cv2.getTrackbarPos('Min Area', window)
            self.params['max_area'] = max(100, cv2.getTrackbarPos('Max Area', window))
            self.params['min_aspect'] = cv2.getTrackbarPos('Min Aspect x100', window)
            self.params['max_aspect'] = max(50, cv2.getTrackbarPos('Max Aspect x100', window))
            
            # Ensure odd numbers for kernel sizes
            if self.params['blur_kernel'] % 2 == 0:
                self.params['blur_kernel'] += 1
            if self.params['thresh_block'] % 2 == 0:
                self.params['thresh_block'] += 1
            if self.params['morph_kernel'] % 2 == 0:
                self.params['morph_kernel'] += 1
            if self.params['final_blur_kernel'] % 2 == 0:
                self.params['final_blur_kernel'] += 1
                
        except cv2.error:
            # Trackbars don't exist yet, use current parameter values
            print("Trackbars not ready yet, using current parameter values")
            pass
        except Exception as e:
            print(f"Error updating parameters: {e}")
            pass
    
    def process_image(self):
        """Process image with current parameters and update displays"""
        # Step 0: Apply brightness adjustment
        if self.params['brightness'] != 100:
            # Brightness adjustment: 100 = normal, 0 = very dark, 200 = very bright
            brightness_factor = self.params['brightness'] / 100.0
            adjusted_img = cv2.convertScaleAbs(self.original_img, alpha=brightness_factor, beta=0)
        else:
            adjusted_img = self.original_img.copy()
        
        # Extract ROI if selected using brush mask
        if self.use_roi and self.roi_mask is not None:
            # Apply mask to the adjusted image
            roi_img = adjusted_img.copy()
            # Create 3-channel mask for color image
            mask_3channel = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2BGR)
            # Apply mask (keep painted areas, make others black)
            roi_img = cv2.bitwise_and(roi_img, mask_3channel)
            roi_offset = (0, 0)  # No offset needed for mask-based ROI
        else:
            roi_img = adjusted_img
            roi_offset = (0, 0)
        
        # Step 1: Convert to grayscale and blur (work on ROI)
        self.gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        if self.params['blur_kernel'] > 1:
            self.gray = cv2.GaussianBlur(self.gray, 
                                       (self.params['blur_kernel'], self.params['blur_kernel']), 
                                       self.params['blur_sigma'])
        
        # Step 2: Adaptive threshold
        self.thresh = cv2.adaptiveThreshold(
            self.gray,
            self.params['thresh_max'],
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.params['thresh_block'],
            -self.params['thresh_c']  # Negative C value as in original
        )
        
        # Step 3: Morphological operations
        if self.params['morph_kernel'] > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.params['morph_kernel'], self.params['morph_kernel']))
            
            if self.params['erode_iter'] > 0:
                self.thresh = cv2.erode(self.thresh, kernel, iterations=self.params['erode_iter'])
            
            if self.params['dilate_iter'] > 0:
                self.thresh = cv2.dilate(self.thresh, kernel, iterations=self.params['dilate_iter'])
        
        # Final blur
        if self.params['final_blur_kernel'] > 1:
            self.thresh = cv2.GaussianBlur(self.thresh, 
                                         (self.params['final_blur_kernel'], self.params['final_blur_kernel']), 
                                         self.params['final_blur_sigma'])
        
        # Step 4: Find and filter contours
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        filtered_contours = []
        min_area = self.params['min_area']
        max_area = self.params['max_area']
        min_aspect_ratio = self.params['min_aspect'] / 100.0  # Convert from trackbar scale
        max_aspect_ratio = self.params['max_aspect'] / 100.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            if (area >= min_area and area <= max_area and 
                aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio):
                filtered_contours.append(contour)
        
        # Update statistics
        self.stats['total_contours'] = len(contours)
        self.stats['filtered_contours'] = len(filtered_contours)
        self.stats['detection_ratio'] = len(filtered_contours) / len(contours) if contours else 0
        
        # Step 5: Draw results on original full image
        self.result_img = self.original_img.copy()
        
        # Apply brightness to display image too
        if self.params['brightness'] != 100:
            brightness_factor = self.params['brightness'] / 100.0
            self.result_img = cv2.convertScaleAbs(self.result_img, alpha=brightness_factor, beta=0)
        
        # Draw ROI boundary if used (brush mask overlay)
        if self.use_roi and self.roi_mask is not None:
            # Create colored overlay for painted areas
            colored_mask = np.zeros_like(self.result_img)
            colored_mask[self.roi_mask > 0] = [255, 0, 255]  # Purple for painted areas
            # Blend with result image
            self.result_img = cv2.addWeighted(self.result_img, 0.85, colored_mask, 0.15, 0)
            
            # Draw brush boundary outline
            contours_mask, _ = cv2.findContours(self.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.result_img, contours_mask, -1, (255, 0, 255), 2)
            
            # Add ROI label
            if contours_mask:
                # Find topmost point of largest contour for label placement
                largest_contour = max(contours_mask, key=cv2.contourArea)
                topmost = tuple(largest_contour[largest_contour[:,:,1].argmin()][0])
                cv2.putText(self.result_img, 'PAINTED ROI', (topmost[0], topmost[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Adjust contour coordinates for full image display (no offset needed for mask-based ROI)
        offset_x, offset_y = roi_offset
        
        # Draw all contours in red
        cv2.drawContours(self.result_img, contours, -1, (0, 0, 255), 1)
        
        # Draw filtered contours in green and centers
        for i, contour in enumerate(filtered_contours):
            # Compute center of mass (no offset adjustment needed for mask-based ROI)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Draw center point
                cv2.circle(self.result_img, (cx, cy), 5, (255, 255, 0), -1)
                
                # Draw chicken number
                cv2.putText(self.result_img, str(i+1), (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw area info
                area = cv2.contourArea(contour)
                cv2.putText(self.result_img, f"{int(area)}", (cx+10, cy+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw filtered contours
        cv2.drawContours(self.result_img, filtered_contours, -1, (0, 255, 0), 2)
        
        # Add statistics overlay
        self.add_statistics_overlay()
        
        # Update all displays
        self.update_displays()
    
    def add_statistics_overlay(self):
        """Add parameter and statistics overlay to result image"""
        # Background for text
        overlay = self.result_img.copy()
        h, w = overlay.shape[:2]
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.result_img, 0.3, 0, self.result_img)
        
        # Statistics text
        roi_info = ""
        if self.use_roi and self.roi_mask is not None:
            painted_pixels = np.sum(self.roi_mask > 0)
            total_pixels = self.roi_mask.shape[0] * self.roi_mask.shape[1]
            coverage_percent = (painted_pixels / total_pixels) * 100
            roi_info = f"ROI: {coverage_percent:.1f}% painted ({painted_pixels} pixels)"
        else:
            roi_info = "ROI: Full image"
        
        text_lines = [
            f"CHICKEN DETECTION STATS",
            f"Total Contours: {self.stats['total_contours']}",
            f"Chickens Detected: {self.stats['filtered_contours']}",
            f"Detection Ratio: {self.stats['detection_ratio']:.2%}",
            f"",
            roi_info,
            f"",
            f"CURRENT PARAMETERS:",
            f"Brightness: {self.params['brightness']}% (100=normal)",
            f"Blur: {self.params['blur_kernel']}x{self.params['blur_kernel']}, σ={self.params['blur_sigma']}",
            f"Threshold: max={self.params['thresh_max']}, block={self.params['thresh_block']}, C=-{self.params['thresh_c']}",
            f"Morphology: {self.params['morph_kernel']}x{self.params['morph_kernel']}, E={self.params['erode_iter']}, D={self.params['dilate_iter']}",
            f"Area: {self.params['min_area']}-{self.params['max_area']}",
            f"Aspect: {self.params['min_aspect']/100:.2f}-{self.params['max_aspect']/100:.2f}"
        ]
        
        y_offset = 35
        for line in text_lines:
            color = (0, 255, 255) if line.startswith(("CHICKEN", "CURRENT")) else (255, 255, 255)
            if "Detected:" in line:
                color = (0, 255, 0)
            
            cv2.putText(self.result_img, line, (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    def update_displays(self):
        """Update all display windows"""
        # Show processing steps
        cv2.imshow('1. Grayscale + Blur', self.gray)
        cv2.imshow('2. Threshold', self.thresh)
        
        # Create morphological result for display
        morph_display = self.thresh.copy()
        cv2.imshow('3. Morphological', morph_display)
        
        # Show main result
        cv2.imshow('Chicken Detection - Real-time Tuning', self.result_img)
    
    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        defaults = {
            'brightness': 100,
            'blur_kernel': 3,
            'blur_sigma': 2,
            'thresh_max': 255,
            'thresh_block': 85,
            'thresh_c': 21,
            'morph_kernel': 9,
            'erode_iter': 2,
            'dilate_iter': 1,
            'final_blur_kernel': 3,
            'final_blur_sigma': 1,
            'min_area': 500,
            'max_area': 10000,
            'min_aspect': 30,
            'max_aspect': 300,
        }
        
        window = 'Control Panel'
        for param, value in defaults.items():
            self.params[param] = value
            # Update trackbar positions
            if param == 'brightness':
                cv2.setTrackbarPos('Brightness', window, value)
            elif param == 'blur_kernel':
                cv2.setTrackbarPos('Blur Kernel', window, value)
            elif param == 'blur_sigma':
                cv2.setTrackbarPos('Blur Sigma', window, value)
            elif param == 'thresh_max':
                cv2.setTrackbarPos('Threshold Max', window, value)
            elif param == 'thresh_block':
                cv2.setTrackbarPos('Threshold Block', window, value)
            elif param == 'thresh_c':
                cv2.setTrackbarPos('Threshold C', window, value)
            elif param == 'morph_kernel':
                cv2.setTrackbarPos('Morph Kernel', window, value)
            elif param == 'erode_iter':
                cv2.setTrackbarPos('Erode Iterations', window, value)
            elif param == 'dilate_iter':
                cv2.setTrackbarPos('Dilate Iterations', window, value)
            elif param == 'final_blur_kernel':
                cv2.setTrackbarPos('Final Blur Kernel', window, value)
            elif param == 'final_blur_sigma':
                cv2.setTrackbarPos('Final Blur Sigma', window, value)
            elif param == 'min_area':
                cv2.setTrackbarPos('Min Area', window, value)
            elif param == 'max_area':
                cv2.setTrackbarPos('Max Area', window, value)
            elif param == 'min_aspect':
                cv2.setTrackbarPos('Min Aspect x100', window, value)
            elif param == 'max_aspect':
                cv2.setTrackbarPos('Max Aspect x100', window, value)
        
        self.process_image()
        print("✅ Parameters reset to defaults")
    
    def save_parameters(self):
        """Save current parameters to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"chicken_detection_params_{timestamp}.json"
        
        save_data = {
            'parameters': self.params.copy(),
            'roi_mask': self.roi_mask.tolist() if self.roi_mask is not None else None,
            'use_roi': self.use_roi,
            'brush_size': self.brush_size,
            'statistics': self.stats.copy(),
            'timestamp': timestamp,
            'notes': 'Parameters tuned using interactive interface with brush ROI'
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Also save ROI mask as image if it exists
        if self.roi_mask is not None:
            mask_filename = filename.replace('.json', '_roi_mask.png')
            cv2.imwrite(mask_filename, self.roi_mask)
            print(f"✓ ROI mask saved to: {mask_filename}")
        
        print(f"✓ Parameters saved to: {filename}")
        return filename
    
    def load_parameters(self, filename=None):
        """Load parameters from JSON file"""
        if filename is None:
            # Try to find the most recent parameter file
            import glob
            param_files = glob.glob("chicken_detection_params_*.json")
            if not param_files:
                print("No parameter files found")
                return False
            
            # Get most recent file
            filename = max(param_files, key=lambda x: os.path.getctime(x))
            print(f"Loading most recent parameter file: {filename}")
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if 'parameters' in data:
                # Update parameters
                for param, value in data['parameters'].items():
                    if param in self.params:
                        self.params[param] = value
                
                # Update ROI settings if available
                if 'roi_mask' in data and data['roi_mask'] is not None:
                    self.roi_mask = np.array(data['roi_mask'], dtype=np.uint8)
                if 'use_roi' in data:
                    self.use_roi = data['use_roi']
                if 'brush_size' in data:
                    self.brush_size = data['brush_size']
                
                # Update trackbars
                window = 'Control Panel'
                cv2.setTrackbarPos('Brightness', window, self.params.get('brightness', 100))
                cv2.setTrackbarPos('Blur Kernel', window, self.params['blur_kernel'])
                cv2.setTrackbarPos('Blur Sigma', window, self.params['blur_sigma'])
                cv2.setTrackbarPos('Threshold Max', window, self.params['thresh_max'])
                cv2.setTrackbarPos('Threshold Block', window, self.params['thresh_block'])
                cv2.setTrackbarPos('Threshold C', window, self.params['thresh_c'])
                cv2.setTrackbarPos('Morph Kernel', window, self.params['morph_kernel'])
                cv2.setTrackbarPos('Erode Iterations', window, self.params['erode_iter'])
                cv2.setTrackbarPos('Dilate Iterations', window, self.params['dilate_iter'])
                cv2.setTrackbarPos('Final Blur Kernel', window, self.params['final_blur_kernel'])
                cv2.setTrackbarPos('Final Blur Sigma', window, self.params['final_blur_sigma'])
                cv2.setTrackbarPos('Min Area', window, self.params['min_area'])
                cv2.setTrackbarPos('Max Area', window, self.params['max_area'])
                cv2.setTrackbarPos('Min Aspect x100', window, self.params['min_aspect'])
                cv2.setTrackbarPos('Max Aspect x100', window, self.params['max_aspect'])
                
                self.process_image()
                print(f"✓ Parameters loaded from: {filename}")
                return True
        
        except Exception as e:
            print(f"❌ Error loading parameters: {e}")
            return False
    
    def print_current_parameters(self):
        """Print current parameters in code format"""
        print("\n" + "="*50)
        print("CURRENT PARAMETERS (copy to your code):")
        print("="*50)
        
        print("# Load and adjust brightness")
        print("img = cv2.imread('your_image.jpg')")
        if self.params['brightness'] != 100:
            brightness_factor = self.params['brightness'] / 100.0
            print(f"img = cv2.convertScaleAbs(img, alpha={brightness_factor:.2f}, beta=0)  # Brightness adjustment")
        
        # Add ROI extraction code if ROI is used
        if self.use_roi and self.roi_mask is not None:
            print(f"\n# Apply brush-painted ROI mask")
            print(f"# Note: You'll need to save and load the ROI mask separately")
            print(f"# roi_mask = cv2.imread('roi_mask.png', cv2.IMREAD_GRAYSCALE)  # Load your painted mask")
            print(f"# Create 3-channel mask")
            print(f"mask_3channel = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)")
            print(f"# Apply mask to image")
            print(f"img = cv2.bitwise_and(img, mask_3channel)")
        
        print("\n# Convert to grayscale")
        print("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        
        print("\n# Gaussian blur")
        print(f"gray = cv2.GaussianBlur(gray, ({self.params['blur_kernel']},{self.params['blur_kernel']}), {self.params['blur_sigma']})")
        
        print("\n# Adaptive threshold")
        print(f"thresh = cv2.adaptiveThreshold(gray, {self.params['thresh_max']}, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, {self.params['thresh_block']}, -{self.params['thresh_c']})")
        
        print("\n# Morphological operations")
        print(f"kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({self.params['morph_kernel']},{self.params['morph_kernel']}))")
        print(f"thresh = cv2.erode(thresh, kernel, iterations={self.params['erode_iter']})")
        print(f"thresh = cv2.dilate(thresh, kernel, iterations={self.params['dilate_iter']})")
        print(f"thresh = cv2.GaussianBlur(thresh, ({self.params['final_blur_kernel']},{self.params['final_blur_kernel']}), {self.params['final_blur_sigma']})")
        
        print("\n# Contour filtering")
        print(f"min_area = {self.params['min_area']}")
        print(f"max_area = {self.params['max_area']}")
        print(f"min_aspect_ratio = {self.params['min_aspect']/100:.2f}")
        print(f"max_aspect_ratio = {self.params['max_aspect']/100:.2f}")
        
        print(f"\n# Results: {self.stats['filtered_contours']} chickens detected")
        if self.use_roi and self.roi_mask is not None:
            painted_pixels = np.sum(self.roi_mask > 0)
            total_pixels = self.roi_mask.shape[0] * self.roi_mask.shape[1]
            coverage = (painted_pixels / total_pixels) * 100
            print(f"# Note: Detection was performed on {coverage:.1f}% of image ({painted_pixels} pixels)")
            print(f"# Coordinates are already in full image space (no offset needed)")
        print("="*50)
    
    def run(self):
        """Main loop for interactive tuning"""
        print("\n🎨 INTERACTIVE CHICKEN DETECTION TUNER")
        print("="*45)
        print("Use brush painting to select detection areas")
        print("Adjust the sliders in the Control Panel to tune detection")
        print("Watch the results update in real-time!")
        print("\nKeyboard Controls:")
        print("  'r' - Reset to defaults")
        print("  's' - Save current parameters")
        print("  'l' - Load saved parameters")
        print("  'p' - Print current parameters")
        print("  'q' - Quit")
        print("\nTuning Tips:")
        print("1. Start with threshold parameters (block size, C value)")
        print("2. Adjust morphological operations to clean up noise")
        print("3. Fine-tune contour filtering (area and aspect ratio)")
        print("4. Watch the 'Chickens Detected' count in real-time")
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Resetting parameters...")
                self.reset_to_defaults()
            elif key == ord('s'):
                print("Saving parameters...")
                self.save_parameters()
            elif key == ord('l'):
                print("Loading parameters...")
                self.load_parameters()
            elif key == ord('o'):
                print("Reselecting ROI area...")
                # Hide current windows temporarily
                cv2.destroyAllWindows()
                # Reselect ROI
                if self.select_roi():
                    # Recreate windows and reprocess
                    self.setup_windows()
                    self.create_trackbars()
                    # Restore all trackbar positions
                    self.restore_trackbar_positions()
                    self.process_image()
                else:
                    print("ROI reselection cancelled")
                    break
            elif key == ord('p'):
                self.print_current_parameters()
            elif key == 27:  # Escape key
                break
        
        cv2.destroyAllWindows()
        
        # Final summary
        print(f"\n🎉 FINAL RESULTS:")
        print(f"Chickens detected: {self.stats['filtered_contours']}")
        print(f"Total contours: {self.stats['total_contours']}")
        print(f"Detection efficiency: {self.stats['detection_ratio']:.2%}")

def main():
    """Main function to start the interactive tuner"""
    # Change this to your image path
    image_path = 'test-data/REALISTIC-PHOTO.png'
    
    try:
        # Create and run the tuner
        print(f"Loading image: {image_path}")
        
        # Check if image exists
        import os
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            print("Please check the file path and make sure the image exists.")
            return
        
        tuner = InteractiveChickenTuner(image_path)
        tuner.run()
        
    except ValueError as e:
        if "ROI selection cancelled" in str(e):
            print("ROI selection was cancelled. Exiting.")
        else:
            print(f"❌ Error: {e}")
            print("Make sure your image path is correct and the image exists.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("This might be due to:")
        print("1. Image file not found or corrupted")
        print("2. OpenCV installation issues")
        print("3. Insufficient system resources")
        print("\nTry:")
        print("- Check the image path is correct")
        print("- Make sure the image file exists")
        print("- Restart the program")
        
        import traceback
        print(f"\nFull error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()