import cv2
import numpy as np
import os
from datetime import datetime

class InteractiveROISelector:
    def __init__(self, background_path, foreground_path):
        # Load images
        self.background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        self.foreground = cv2.imread(foreground_path, cv2.IMREAD_GRAYSCALE)
        
        if self.background is None or self.foreground is None:
            raise ValueError("Could not load images. Check file paths!")
        
        # Create mask (white = analyze, black = ignore)
        self.mask = np.zeros(self.background.shape, dtype=np.uint8)
        
        # Brush parameters
        self.brush_size = 20
        self.drawing = False
        self.erasing = False
        
        # Display image for ROI selection (use foreground with chickens)
        self.display_image = cv2.cvtColor(self.foreground, cv2.COLOR_GRAY2BGR)
        self.roi_overlay = self.display_image.copy()
        
        # Setup windows and mouse callback
        cv2.namedWindow('ROI Selection', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('ROI Selection', self.mouse_callback)
        
        # Create trackbar for brush size
        cv2.createTrackbar('Brush Size', 'Controls', self.brush_size, 100, self.update_brush_size)
        
        self.update_display()
        
    def update_brush_size(self, val):
        """Update brush size from trackbar"""
        self.brush_size = max(1, val)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.erasing = False
            self.paint_roi(x, y)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erasing = True
            self.paint_roi(x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.paint_roi(x, y)
                
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            
        self.update_display()
    
    def paint_roi(self, x, y):
        """Paint or erase ROI at given coordinates"""
        color = 0 if self.erasing else 255
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
    
    def update_display(self):
        """Update the display with current ROI selection"""
        # Create overlay showing selected region
        self.roi_overlay = self.display_image.copy()
        
        # Show selected areas in semi-transparent green
        selected_areas = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        selected_areas[:, :, 0] = 0  # Remove blue channel
        selected_areas[:, :, 2] = 0  # Remove red channel
        
        # Blend with original image
        self.roi_overlay = cv2.addWeighted(self.roi_overlay, 0.7, selected_areas, 0.3, 0)
        
        # Add instructions text
        cv2.putText(self.roi_overlay, 'LEFT CLICK: Paint ROI', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.roi_overlay, 'RIGHT CLICK: Erase', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.roi_overlay, 'C: Clear all, R: Reset, SPACE: Continue', (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.roi_overlay, f'Brush Size: {self.brush_size}', (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('ROI Selection', self.roi_overlay)
    
    def select_roi(self):
        """Interactive ROI selection interface"""
        print("ROI Selection Interface")
        print("=" * 40)
        print("LEFT CLICK + DRAG: Paint areas to analyze (green)")
        print("RIGHT CLICK + DRAG: Erase areas") 
        print("Adjust 'Brush Size' slider to change brush size")
        print("Press 'c' to clear all")
        print("Press 'r' to reset to full image")
        print("Press SPACE when done selecting")
        print("Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord(' '):  # Space - continue with analysis
                break
            elif key == ord('c'):  # Clear all
                self.mask = np.zeros(self.background.shape, dtype=np.uint8)
                self.update_display()
                print("ROI cleared")
            elif key == ord('r'):  # Reset to full image
                self.mask = np.ones(self.background.shape, dtype=np.uint8) * 255
                self.update_display()
                print("ROI reset to full image")
            elif key == ord('q'):  # Quit
                return False
        
        # Check if any ROI is selected
        if np.sum(self.mask) == 0:
            print("No ROI selected! Using full image.")
            self.mask = np.ones(self.background.shape, dtype=np.uint8) * 255
        
        return True
    
    def apply_roi_mask(self, image):
        """Apply ROI mask to an image"""
        return cv2.bitwise_and(image, image, mask=self.mask)
    
    def get_masked_images(self):
        """Get background and foreground with ROI mask applied"""
        masked_background = self.apply_roi_mask(self.background)
        masked_foreground = self.apply_roi_mask(self.foreground)
        return masked_background, masked_foreground, self.mask

class ROIBackgroundSubtractor:
    def __init__(self):
        # Simple cleanup parameter - just remove small noise
        self.min_component_area = 200      # Remove components smaller than this (chickens are large)
        
        # Area-based estimation parameters
        self.single_chicken_areas = []     # Store areas of likely single chickens
        self.average_chicken_area = 1700   # Default estimate, will be updated
        self.max_single_chicken_area = 16000  # Areas above this are likely multiple chickens
        
        # NEW: Chicken tracking for incremental IDs
        self.tracked_chickens = []         # List of tracked chicken objects
        self.next_chicken_id = 1           # Next ID to assign to a new chicken
        self.max_tracking_distance = 150   # Max distance to consider same chicken between frames
        self.frames_before_removal = 10    # Remove chicken if not seen for this many frames
        pass
    
    def remove_small_components(self, binary_mask, min_area=200, debug=False):
        """
        Simple removal of small connected components from binary mask
        
        Args:
            binary_mask: Input binary mask (0 and 255 values)
            min_area: Minimum area to keep (components smaller than this are removed)
            debug: If True, show what was removed
        
        Returns:
            cleaned_mask: Binary mask with small components removed
        """
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Create output mask
        cleaned_mask = np.zeros_like(binary_mask)
        
        # Keep only components larger than min_area (skip label 0 which is background)
        kept_components = 0
        removed_small = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                # Keep this component - it's large enough to potentially be a chicken
                cleaned_mask[labels == i] = 255
                kept_components += 1
            else:
                removed_small += 1
        
        if debug:
            print(f"  Components kept: {kept_components}")
            print(f"  Small components removed: {removed_small}")
            print(f"  Min area threshold: {min_area}")
        
        return cleaned_mask
    
    def method1_simple_difference(self, background, foreground):
        """Simple absolute difference with small component removal"""
        bg_blur = cv2.GaussianBlur(background, (5, 5), 0)
        fg_blur = cv2.GaussianBlur(foreground, (5, 5), 0)
        diff = cv2.absdiff(fg_blur, bg_blur)
        _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Remove small noise components
        binary_cleaned = self.remove_small_components(binary, self.min_component_area)
        
        return diff, binary_cleaned
    
    def method2_adaptive_difference(self, background, foreground):
        """Adaptive thresholding on difference with small component removal"""
        diff = cv2.absdiff(foreground, background)
        binary = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Remove small noise components
        binary_cleaned = self.remove_small_components(binary, self.min_component_area)
        
        return diff, binary_cleaned
    
    def method3_morphological_cleanup(self, background, foreground):
        """Difference with morphological cleanup + small component removal"""
        bg_blur = cv2.GaussianBlur(background, (3, 3), 0)
        fg_blur = cv2.GaussianBlur(foreground, (3, 3), 0)
        diff = cv2.absdiff(fg_blur, bg_blur)
        _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Original morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small noise components
        binary_cleaned = self.remove_small_components(binary, self.min_component_area)
        
        return diff, binary_cleaned
    
    def method4_enhanced_difference(self, background, foreground):
        """Enhanced method with contrast adjustment + small component removal"""
        bg_enhanced = cv2.convertScaleAbs(background, alpha=1.2, beta=10)
        fg_enhanced = cv2.convertScaleAbs(foreground, alpha=1.2, beta=10)
        
        bg_filtered = cv2.bilateralFilter(bg_enhanced, 9, 75, 75)
        fg_filtered = cv2.bilateralFilter(fg_enhanced, 9, 75, 75)
        
        diff = cv2.absdiff(fg_filtered, bg_filtered)
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small noise components
        binary_cleaned = self.remove_small_components(binary, self.min_component_area)
        
        return diff, binary_cleaned
    
    def method5_statistical_difference(self, background, foreground):
        """Statistical approach with small component removal"""
        bg_float = background.astype(np.float32)
        fg_float = foreground.astype(np.float32)
        
        diff = np.abs(fg_float - bg_float)
        threshold = np.mean(diff) + 2 * np.std(diff)
        
        binary = (diff > threshold).astype(np.uint8) * 255
        diff = diff.astype(np.uint8)
        
        # Remove small noise components
        binary_cleaned = self.remove_small_components(binary, self.min_component_area)
        
        return diff, binary_cleaned
    
    class TrackedChicken:
        """Class to represent a tracked chicken with persistent ID"""
        def __init__(self, chicken_id, centroid, area, contour, extreme_points):
            self.id = chicken_id
            self.centroid = centroid
            self.area = area
            self.contour = contour
            self.extreme_points = extreme_points
            self.frames_since_seen = 0
            self.total_detections = 1
            self.first_seen_frame = None
    
    def update_chicken_tracking(self, current_detections, current_frame):
        """
        Update chicken tracking with current frame detections.
        Assigns persistent IDs to chickens across frames.
        
        Args:
            current_detections: List of chicken detection data for current frame
            current_frame: Current frame number
            
        Returns:
            List of tracked chicken data with persistent IDs
        """
        # Mark all existing chickens as not seen this frame
        for chicken in self.tracked_chickens:
            chicken.frames_since_seen += 1
        
        tracked_frame_data = []
        
        # Try to match each detection to existing tracked chickens
        for detection in current_detections:
            detection_centroid = detection['centroid']
            best_match = None
            best_distance = float('inf')
            
            # Find closest existing chicken
            for tracked_chicken in self.tracked_chickens:
                if tracked_chicken.frames_since_seen < self.frames_before_removal:
                    distance = np.sqrt(
                        (detection_centroid[0] - tracked_chicken.centroid[0])**2 + 
                        (detection_centroid[1] - tracked_chicken.centroid[1])**2
                    )
                    
                    if distance < self.max_tracking_distance and distance < best_distance:
                        best_match = tracked_chicken
                        best_distance = distance
            
            if best_match:
                # Update existing chicken
                best_match.centroid = detection_centroid
                best_match.area = detection['area']
                best_match.contour = detection['contour']
                best_match.extreme_points = detection['extreme_points']
                best_match.frames_since_seen = 0
                best_match.total_detections += 1
                
                # Create tracked data for this frame
                tracked_data = detection.copy()
                tracked_data['persistent_id'] = best_match.id
                tracked_data['total_detections'] = best_match.total_detections
                tracked_data['first_seen_frame'] = best_match.first_seen_frame
                tracked_frame_data.append(tracked_data)
                
            else:
                # New chicken detected - assign new persistent ID
                new_chicken = self.TrackedChicken(
                    self.next_chicken_id,
                    detection_centroid,
                    detection['area'],
                    detection['contour'],
                    detection['extreme_points']
                )
                new_chicken.first_seen_frame = current_frame
                
                self.tracked_chickens.append(new_chicken)
                
                # Create tracked data for this frame
                tracked_data = detection.copy()
                tracked_data['persistent_id'] = self.next_chicken_id
                tracked_data['total_detections'] = 1
                tracked_data['first_seen_frame'] = current_frame
                tracked_frame_data.append(tracked_data)
                
                print(f"NEW CHICKEN #{self.next_chicken_id} detected at frame {current_frame}!")
                self.next_chicken_id += 1
        
        # Remove old chickens that haven't been seen for too long
        self.tracked_chickens = [
            chicken for chicken in self.tracked_chickens 
            if chicken.frames_since_seen < self.frames_before_removal
        ]
        
        return tracked_frame_data

    # def find_contours_and_extreme_points(self, binary_image, roi_mask, current_frame=0):
    def find_contours_and_extreme_points(self, binary_image, roi_mask, current_frame=0):
        """
        Find contours and extract extreme points (furthest outward points) from binary mask.
        Now includes chicken tracking with persistent IDs.
        
        Args:
            binary_image: Binary mask of detected chickens
            roi_mask: ROI mask to limit analysis area
            current_frame: Current frame number for tracking
            
        Returns:
            List of dictionaries containing contour data, extreme points, and persistent IDs
        """
        # Apply ROI mask to binary result
        binary_roi = cv2.bitwise_and(binary_image, binary_image, mask=roi_mask)
        
        # Find contours
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out extremely large areas that are clearly not chickens
        max_reasonable_area = 25000
        min_reasonable_area = self.min_component_area
        
        current_detections = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_reasonable_area <= area <= max_reasonable_area:
                # Find extreme points - furthest outward points on the contour
                if len(contour) > 0:
                    # Find the furthest points in each direction
                    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
                    bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
                    
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    else:
                        # Fallback to bounding box center
                        x, y, w, h = cv2.boundingRect(contour)
                        centroid = (x + w//2, y + h//2)
                    
                    # Estimate number of chickens based on area
                    if area <= self.max_single_chicken_area:
                        estimated_chickens = 1
                        self.single_chicken_areas.append(area)
                        if len(self.single_chicken_areas) > 50:
                            self.single_chicken_areas = self.single_chicken_areas[-50:]
                        self.average_chicken_area = sum(self.single_chicken_areas) / len(self.single_chicken_areas)
                    else:
                        estimated_chickens = max(1, round(area / self.average_chicken_area))
                    
                    # Store detection data (without persistent ID yet)
                    detection_info = {
                        'temp_id': i,  # Temporary ID for this frame
                        'contour': contour,
                        'area': area,
                        'centroid': centroid,
                        'estimated_chickens': estimated_chickens,
                        'extreme_points': {
                            'leftmost': leftmost,
                            'rightmost': rightmost,
                            'topmost': topmost,
                            'bottommost': bottommost
                        },
                        'bounding_rect': cv2.boundingRect(contour)
                    }
                    
                    current_detections.append(detection_info)
        
        # Now apply tracking to assign persistent IDs
        tracked_chickens = self.update_chicken_tracking(current_detections, current_frame)
        
        return tracked_chickens
    
    def create_segmentation_overlay(self, original_color_frame, chicken_data, roi_mask, current_frame=0):
        """
        Create segmentation overlay showing chickens with extreme points marked.
        Now shows persistent tracking IDs instead of per-frame counts.
        
        Args:
            original_color_frame: Original color video frame
            chicken_data: List of tracked chicken detection data with persistent IDs
            roi_mask: ROI mask
            current_frame: Current frame number
            
        Returns:
            Color frame with segmentation overlay and total unique chickens seen
        """
        # Create overlay on original frame
        segmentation_frame = original_color_frame.copy()
        
        # Show ROI area with slight blue tint
        roi_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        roi_color[:, :, 0] = roi_mask  # Blue channel
        roi_color[:, :, 1] = 0         # Green channel
        roi_color[:, :, 2] = 0         # Red channel
        segmentation_frame = cv2.addWeighted(segmentation_frame, 0.9, roi_color, 0.1, 0)
        
        total_chickens_in_frame = 0
        
        # Draw each chicken's segmentation and extreme points with persistent IDs
        for chicken in chicken_data:
            contour = chicken['contour']
            extreme_points = chicken['extreme_points']
            centroid = chicken['centroid']
            area = chicken['area']
            estimated_chickens = chicken['estimated_chickens']
            persistent_id = chicken['persistent_id']
            total_detections = chicken['total_detections']
            first_seen = chicken['first_seen_frame']
            
            total_chickens_in_frame += estimated_chickens
            
            # Draw filled contour with transparency
            contour_overlay = np.zeros_like(segmentation_frame)
            if estimated_chickens == 1:
                cv2.fillPoly(contour_overlay, [contour], (0, 255, 0))  # Green for single chicken
                contour_color = (0, 255, 0)
            else:
                cv2.fillPoly(contour_overlay, [contour], (0, 255, 255))  # Yellow for multiple chickens
                contour_color = (0, 255, 255)
            
            # Blend the filled contour
            segmentation_frame = cv2.addWeighted(segmentation_frame, 0.8, contour_overlay, 0.2, 0)
            
            # Draw contour outline
            cv2.drawContours(segmentation_frame, [contour], -1, contour_color, 2)
            
            # Draw extreme points with different colors - these show the furthest outward points
            point_size = 6
            cv2.circle(segmentation_frame, extreme_points['leftmost'], point_size, (255, 0, 0), -1)    # Blue - leftmost
            cv2.circle(segmentation_frame, extreme_points['rightmost'], point_size, (0, 0, 255), -1)   # Red - rightmost
            cv2.circle(segmentation_frame, extreme_points['topmost'], point_size, (255, 255, 0), -1)   # Cyan - topmost
            cv2.circle(segmentation_frame, extreme_points['bottommost'], point_size, (0, 255, 255), -1) # Yellow - bottommost
            
            # Draw centroid
            cv2.circle(segmentation_frame, centroid, 4, (255, 255, 255), -1)  # White center
            
            # Draw bounding box
            x, y, w, h = chicken['bounding_rect']
            cv2.rectangle(segmentation_frame, (x, y), (x + w, y + h), (255, 0, 255), 1)
            
            # Add persistent chicken ID and tracking info
            cv2.putText(segmentation_frame, f'ID #{persistent_id}', 
                       (centroid[0] - 15, centroid[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add detection count for this chicken
            cv2.putText(segmentation_frame, f'Seen: {total_detections}x', 
                       (centroid[0] - 20, centroid[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if estimated_chickens > 1:
                cv2.putText(segmentation_frame, f'{estimated_chickens}ch', 
                           (centroid[0] - 10, centroid[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add title and statistics
        cv2.putText(segmentation_frame, 'Chicken Tracking with Extreme Points', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(segmentation_frame, f'Frame: {current_frame} | In Frame: {total_chickens_in_frame}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(segmentation_frame, f'Total Unique Chickens Seen: {self.next_chicken_id - 1}', (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(segmentation_frame, f'Currently Tracking: {len(self.tracked_chickens)}', (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend for extreme points
        legend_y = 125
        cv2.circle(segmentation_frame, (20, legend_y), 4, (255, 0, 0), -1)
        cv2.putText(segmentation_frame, 'Leftmost', (30, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.circle(segmentation_frame, (120, legend_y), 4, (0, 0, 255), -1)
        cv2.putText(segmentation_frame, 'Rightmost', (130, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.circle(segmentation_frame, (220, legend_y), 4, (255, 255, 0), -1)
        cv2.putText(segmentation_frame, 'Topmost', (230, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.circle(segmentation_frame, (310, legend_y), 4, (0, 255, 255), -1)
        cv2.putText(segmentation_frame, 'Bottommost', (320, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return segmentation_frame, total_chickens_in_frame

    def find_and_count_chickens(self, binary_image, original_image, roi_mask):
        """Find contours and count chickens using area-based estimation for touching chickens"""
        # Apply ROI mask to binary result to ensure we only count in selected areas
        binary_roi = cv2.bitwise_and(binary_image, binary_image, mask=roi_mask)
        
        # Find contours
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter out extremely large areas that are clearly not chickens
        max_reasonable_area = 25000  # Remove very large merged areas
        
        valid_contours = []
        total_estimated_chickens = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max_reasonable_area:  # Small components already filtered out
                valid_contours.append(contour)
                
                # Area-based chicken estimation
                if area <= self.max_single_chicken_area:
                    # Likely a single chicken - store area for learning and count as 1
                    self.single_chicken_areas.append(area)
                    # Keep only recent measurements (last 50)
                    if len(self.single_chicken_areas) > 50:
                        self.single_chicken_areas = self.single_chicken_areas[-50:]
                    
                    # Update average chicken area
                    self.average_chicken_area = sum(self.single_chicken_areas) / len(self.single_chicken_areas)
                    
                    estimated_chickens = 1
                else:
                    # Large blob - estimate multiple chickens
                    estimated_chickens = max(1, round(area / self.average_chicken_area))
                
                total_estimated_chickens += estimated_chickens
        
        # Create result image
        result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if len(original_image.shape) == 2 else original_image.copy()
        
        # Show ROI area in blue tint
        roi_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        roi_color[:, :, 0] = roi_mask  # Blue channel
        roi_color[:, :, 1] = 0        # Green channel
        roi_color[:, :, 2] = 0        # Red channel
        result = cv2.addWeighted(result, 0.9, roi_color, 0.1, 0)
        
        if valid_contours:
            for contour in valid_contours:
                area = cv2.contourArea(contour)
                
                # Determine if single or multiple chickens
                if area <= self.max_single_chicken_area:
                    # Single chicken - green contour
                    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                    estimated_chickens = 1
                    color = (0, 255, 0)  # Green for single
                else:
                    # Multiple chickens - yellow contour
                    cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
                    estimated_chickens = max(1, round(area / self.average_chicken_area))
                    color = (0, 255, 255)  # Yellow for multiple
                
                # Draw bounding box with chicken count
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Add area and estimated count text
                cv2.putText(result, f'{int(area)}px', (x, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(result, f'{estimated_chickens}ch', (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add count and statistics text
        cv2.putText(result, f'Total Chickens: {total_estimated_chickens}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'Avg Single: {int(self.average_chicken_area)}px', (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result, f'Learning: {len(self.single_chicken_areas)} samples', (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result, total_estimated_chickens
    
    def choose_method_interactive(self, background, foreground, roi_mask):
        """Let user choose the best method interactively"""
        methods = [
            ("Method 1: Simple Difference + Small Component Removal", self.method1_simple_difference),
            ("Method 2: Adaptive Difference + Small Component Removal", self.method2_adaptive_difference),
            ("Method 3: Morphological + Small Component Removal", self.method3_morphological_cleanup),
            ("Method 4: Enhanced Difference + Small Component Removal", self.method4_enhanced_difference),
            ("Method 5: Statistical Difference + Small Component Removal", self.method5_statistical_difference)
        ]
        
        print(f"\nMethod Selection - Choose the best performing method")
        print(f"Small component removal threshold: {self.min_component_area} pixels")
        print("=" * 60)
        
        cv2.namedWindow('Method Comparison', cv2.WINDOW_AUTOSIZE)
        
        for i, (name, method_func) in enumerate(methods):
            print(f"\n{name}")
            print("-" * 40)
            
            # Apply method
            diff, binary = method_func(background, foreground)
            
            # Find and count chickens
            result_with_contours, chicken_count = self.find_and_count_chickens(binary, foreground, roi_mask)
            
            print(f"Chickens detected: {chicken_count}")
            
            # Create display
            diff_display = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            binary_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # Resize for display
            height = 300
            aspect_ratio = diff.shape[1] / diff.shape[0]
            width = int(height * aspect_ratio)
            
            diff_resized = cv2.resize(diff_display, (width, height))
            binary_resized = cv2.resize(binary_display, (width, height))
            result_resized = cv2.resize(result_with_contours, (width, height))
            
            # Combine views
            combined = np.hstack([diff_resized, binary_resized, result_resized])
            
            # Add labels
            cv2.putText(combined, 'Difference', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, 'Binary (No Small)', (width + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, 'ROI Result', (2*width + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Method Comparison', combined)
            
            print(f"Press '{i+1}' to select this method, any other key to continue...")
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(str(i+1)):
                cv2.destroyAllWindows()
                print(f"✓ Selected: {name}")
                return method_func, name
        
        cv2.destroyAllWindows()
        # Default to method 1 if none selected
        print("No method selected, defaulting to Method 1: Simple Difference + Small Component Removal")
        return methods[0][1], methods[0][0]

class VideoChickenProcessor:
    def __init__(self, background_image_path):
        self.background_image = cv2.imread(background_image_path, cv2.IMREAD_GRAYSCALE)
        if self.background_image is None:
            raise ValueError(f"Could not load background image: {background_image_path}")
        
        self.roi_mask = None
        self.chosen_method = None
        self.method_name = ""
        self.processor = ROIBackgroundSubtractor()
        
        # Statistics tracking
        self.frame_count = 0
        self.total_chickens = 0
        self.chicken_counts = []
    
    def setup_roi_and_method(self, reference_image_path):
        """Setup ROI and choose method using reference image"""
        print("Step 1: Setting up ROI and Method Selection")
        print("=" * 50)
        
        # ROI Selection
        roi_selector = InteractiveROISelector(self.background_image_path, reference_image_path)
        if not roi_selector.select_roi():
            return False
        
        # Get masked images and ROI
        masked_background, masked_foreground, self.roi_mask = roi_selector.get_masked_images()
        
        # Method selection
        self.chosen_method, self.method_name = self.processor.choose_method_interactive(
            masked_background, masked_foreground, self.roi_mask
        )
        
        return True
    
    def adjust_tracking_parameters(self):
        """Interactive adjustment of chicken tracking parameters"""
        print("\nAdjust Chicken Tracking Parameters")
        print("=" * 40)
        print(f"Current max_tracking_distance: {self.processor.max_tracking_distance}")
        print("(Maximum pixel distance to consider same chicken between frames)")
        print(f"Current frames_before_removal: {self.processor.frames_before_removal}")
        print("(Remove chicken from tracking if not seen for this many frames)")
        
        try:
            new_distance = input(f"New max_tracking_distance (current: {self.processor.max_tracking_distance}): ").strip()
            if new_distance:
                self.processor.max_tracking_distance = int(new_distance)
                print(f"Updated tracking distance to: {self.processor.max_tracking_distance}")
            
            new_frames = input(f"New frames_before_removal (current: {self.processor.frames_before_removal}): ").strip()
            if new_frames:
                self.processor.frames_before_removal = int(new_frames)
                print(f"Updated frames before removal to: {self.processor.frames_before_removal}")
                
        except ValueError:
            print("Invalid input, keeping current values")
        
        print(f"Tracking: distance={self.processor.max_tracking_distance}, removal={self.processor.frames_before_removal}")

    def adjust_cleanup_parameters(self):
        """Interactive adjustment of small component removal threshold"""
        print("\nAdjust Small Component Removal Threshold")
        print("=" * 40)
        print(f"Current min_component_area: {self.processor.min_component_area}")
        print("(Components smaller than this will be removed as noise)")
        
        try:
            new_min = input(f"New min_component_area (current: {self.processor.min_component_area}): ").strip()
            if new_min:
                self.processor.min_component_area = int(new_min)
                print(f"Updated to: {self.processor.min_component_area}")
            else:
                print("Keeping current value")
                
        except ValueError:
            print("Invalid input, keeping current value")
        
        print(f"Small components below {self.processor.min_component_area} pixels will be removed")
    
    def process_video(self, video_path, output_path=None, show_realtime=True, frame_delay=500):
        """Process entire video with selected ROI and method
        
        Args:
            frame_delay: Delay in milliseconds between frames (500 = half second)
                        Set to 1 for normal speed, 1000 for 1 second delay
        """
        print(f"\nStep 2: Processing Video")
        print("=" * 30)
        print(f"Video: {video_path}")
        print(f"Method: {self.method_name}")
        print(f"Frame delay: {frame_delay}ms")
        print(f"Output: {output_path if output_path else 'Display only'}")
        print(f"Cleanup: removing components smaller than {self.processor.min_component_area} pixels")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if saving
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup display windows - NOW INCLUDING THE NEW SEGMENTATION WINDOW
        if show_realtime:
            cv2.namedWindow('Original Video', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Chicken Detection', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Chicken Segmentation', cv2.WINDOW_AUTOSIZE)  # NEW WINDOW
            cv2.namedWindow('Binary Mask (Small Components Removed)', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Color Difference', cv2.WINDOW_AUTOSIZE)
            
            # Position windows for better layout
            cv2.moveWindow('Original Video', 50, 50)
            cv2.moveWindow('Chicken Segmentation', 650, 50)  # NEW WINDOW position
            cv2.moveWindow('Chicken Detection', 1250, 50)
            cv2.moveWindow('Binary Mask (Small Components Removed)', 50, 450)
            cv2.moveWindow('Color Difference', 650, 450)
        
        # Process frames
        self.frame_count = 0
        self.total_chickens = 0
        self.chicken_counts = []
        
        print("\nProcessing frames...")
        print("Windows: Original | Segmentation (WITH TRACKING) | Detection | Binary Mask | Color Difference")
        print("Controls: SPACE=pause, 'q'=quit, 's'=save frame, '+'=speed up, '-'=slow down")
        print("          'p'=adjust small component threshold, 't'=adjust tracking distance")
        print("NOTE: Each new chicken gets a unique incremental ID (ID #1, ID #2, etc.)")
        print("      'Total Unique Chickens Seen' shows cumulative count of different chickens")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video - rewind and continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                print("Video ended - rewinding...")
                continue
            
            self.frame_count += 1
            
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply ROI mask
            masked_background = cv2.bitwise_and(self.background_image, self.background_image, mask=self.roi_mask)
            masked_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=self.roi_mask)
            
            # Apply chosen method (now returns cleaned binary)
            diff, binary_cleaned = self.chosen_method(masked_background, masked_frame)
            
            # NEW: Find contours and extreme points for segmentation WITH TRACKING
            chicken_data = self.processor.find_contours_and_extreme_points(binary_cleaned, self.roi_mask, self.frame_count)
            
            # NEW: Create segmentation overlay with extreme points AND PERSISTENT IDs
            segmentation_frame, segmentation_chicken_count = self.processor.create_segmentation_overlay(frame, chicken_data, self.roi_mask, self.frame_count)
            
            # Create color difference visualization
            color_diff = self.create_color_difference_visualization(frame, binary_cleaned, self.roi_mask)
            
            # Original: Find and count chickens for the detection window
            result_frame, chicken_count = self.processor.find_and_count_chickens(binary_cleaned, gray_frame, self.roi_mask)
            
            # Update statistics (use segmentation count as it's more detailed)
            self.chicken_counts.append(segmentation_chicken_count)
            self.total_chickens += segmentation_chicken_count
            
            # Add frame info to all windows
            frame_info = f'Frame: {self.frame_count}'
            method_info = f'Method: {self.method_name}'
            
            # Original video frame info
            cv2.putText(frame, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, method_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Detection frame info
            cv2.putText(result_frame, frame_info, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, method_info, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show progress occasionally
            if self.frame_count % 1 == 0:
                avg_chickens = np.mean(self.chicken_counts[-10:]) if len(self.chicken_counts) >= 10 else np.mean(self.chicken_counts) if self.chicken_counts else 0
                print(f"Frame {self.frame_count}: {segmentation_chicken_count} chickens (avg last 10: {avg_chickens:.1f})")
            
            # Display frames - NOW INCLUDING THE NEW SEGMENTATION WINDOW
            if show_realtime:
                cv2.imshow('Original Video', frame)
                cv2.imshow('Chicken Segmentation', segmentation_frame)  # NEW WINDOW with extreme points
                cv2.imshow('Chicken Detection', result_frame)
                cv2.imshow('Binary Mask (Small Components Removed)', cv2.cvtColor(binary_cleaned, cv2.COLOR_GRAY2BGR))
                cv2.imshow('Color Difference', color_diff)
            
            # Save frame if writer is available (save the segmentation frame as it's most informative)
            if writer:
                writer.write(segmentation_frame)
            
            # Handle controls with configurable delay
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Pause
                print("Paused - press any key to continue")
                cv2.waitKey(0)
            elif key == ord('s') and show_realtime:  # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Save both original detection and new segmentation frames
                cv2.imwrite(f"chicken_detection_{timestamp}.jpg", result_frame)
                cv2.imwrite(f"chicken_segmentation_{timestamp}.jpg", segmentation_frame)
                print(f"Saved frames as chicken_detection_{timestamp}.jpg and chicken_segmentation_{timestamp}.jpg")
            elif key == ord('p'):  # Adjust parameters
                print("Pausing to adjust small component removal threshold...")
                self.adjust_cleanup_parameters()
                print("Threshold updated. Processing continues...")
            elif key == ord('t'):  # Adjust tracking parameters
                print("Pausing to adjust chicken tracking parameters...")
                self.adjust_tracking_parameters()
                print("Tracking parameters updated. Processing continues...")
            elif key == ord('+') or key == ord('='):  # Speed up
                frame_delay = max(1, frame_delay - 100)
                print(f"Frame delay: {frame_delay}ms (faster)")
            elif key == ord('-'):  # Slow down
                frame_delay = min(5000, frame_delay + 100)
                print(f"Frame delay: {frame_delay}ms (slower)")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_realtime:
            cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_statistics()
    
    def create_color_difference_visualization(self, original_color_frame, binary_mask, roi_mask):
        """Create a color visualization showing what was detected as different"""
        # Create a copy of the original frame
        color_diff = original_color_frame.copy()
        
        # Create a mask that combines ROI and detection
        combined_mask = cv2.bitwise_and(binary_mask, roi_mask)
        
        # Show detected areas in original colors, darken the rest
        inverse_mask = cv2.bitwise_not(combined_mask)
        
        # Create darkened background
        darkened = color_diff.copy()
        darkened = cv2.convertScaleAbs(darkened, alpha=0.3, beta=0)
        
        # Apply darkening to non-detected areas
        for i in range(3):
            color_diff[:, :, i] = np.where(inverse_mask > 0, darkened[:, :, i], color_diff[:, :, i])
        
        # Add colored highlighting to detected regions
        highlight_overlay = np.zeros_like(color_diff)
        highlight_overlay[:, :, 1] = combined_mask  # Green channel for detected areas
        
        # Blend the highlight with the image
        color_diff = cv2.addWeighted(color_diff, 0.8, highlight_overlay, 0.2, 0)
        
        # Draw ROI boundary in blue
        roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_diff, roi_contours, -1, (255, 0, 0), 2)
        
        # Add labels
        cv2.putText(color_diff, 'Color Difference (Small Components Removed)', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(color_diff, 'Green=Detected, Blue=ROI', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return color_diff
    
    def print_statistics(self):
        """Print processing statistics"""
        if not self.chicken_counts:
            return
            
        print(f"\nVideo Processing Complete!")
        print("=" * 40)
        print(f"Total frames processed: {len(self.chicken_counts)}")
        print(f"Total chickens detected: {sum(self.chicken_counts)}")
        print(f"Average chickens per frame: {np.mean(self.chicken_counts):.1f}")
        print(f"Maximum chickens in frame: {max(self.chicken_counts)}")
        print(f"Minimum chickens in frame: {min(self.chicken_counts)}")
        print(f"Standard deviation: {np.std(self.chicken_counts):.1f}")

def main():
    # Configuration - UPDATE THESE PATHS
    background_image_path = '../../test-data/toremove.png'      # Your background image
    reference_image_path = '../../test-data/nottoremove.png'    # Reference image for ROI/method selection
    video_path = '../../test-data/REALISTIC-RECORDING.mp4'            # Your input video
    output_video_path = '../../output/approach2.mp4'  # Optional: save processed video
    
    try:
        print("Enhanced Chicken Detection with Tracking & Segmentation Window")
        print("=" * 60)
        print("NEW: Chickens now get unique incremental IDs (#1, #2, #3, etc.)")
        print("NEW: Each chicken is tracked across frames with persistent identity!")
        print("NEW: Shows total unique chickens seen vs chickens currently in frame")
        
        # Initialize processor
        processor = VideoChickenProcessor(background_image_path)
        processor.background_image_path = background_image_path
        
        # Setup ROI and method
        if not processor.setup_roi_and_method(reference_image_path):
            print("Setup cancelled")
            return
        
        # Process video with enhanced segmentation window
        processor.process_video(
            video_path=video_path,
            output_path=output_video_path,
            show_realtime=True,
            frame_delay=20
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()