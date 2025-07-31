import cv2
import numpy as np

class ImageBrushRemover:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.image = self.original_image.copy()
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.brush_size = 20
        
        # Create window
        cv2.namedWindow('Image Editor', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Image Editor', self.mouse_callback)
        
        print("Instructions:")
        print("- Left click and drag to mark areas for removal")
        print("- Press 'r' to reset")
        print("- Press 'u' to undo last stroke")
        print("- Press '+' to increase brush size")
        print("- Press '-' to decrease brush size")
        print("- Press 's' to save result to output.png")
        print("- Press 'q' or ESC to quit")
        
    def mouse_callback(self, event, x, y, flags, param):
        self.last_mouse_pos = (x, y)  # Track mouse position
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_mask = self.mask.copy()  # Save state for undo
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def update_display(self):
        # Create display image with red overlay for selected areas
        display_image = self.original_image.copy()
        
        # Apply red overlay where mask is active
        overlay_area = self.mask > 0
        if np.any(overlay_area):
            # Simple blending: mix original with red
            display_image[overlay_area, 2] = np.clip(
                display_image[overlay_area, 2] * 0.7 + 255 * 0.3, 0, 255
            ).astype(np.uint8)
        
        # Draw brush size indicator
        if hasattr(self, 'last_mouse_pos'):
            cv2.circle(display_image, self.last_mouse_pos, self.brush_size, (0, 255, 0), 2)
        
        return display_image
    
    def remove_selected_areas(self):
        """Remove selected areas by replacing with white background"""
        result = self.original_image.copy()
        # Replace selected areas with white
        result[self.mask > 0] = [255, 255, 255]
        return result
    
    def save_result(self):
        result = self.remove_selected_areas()
        cv2.imwrite('../../output/approach6.png', result)
        print("Result saved to output.png")
    
    def run(self):
        while True:
            display_image = self.update_display()
            cv2.imshow('Image Editor', display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                print("Reset complete")
            elif key == ord('u'):  # Undo
                if hasattr(self, 'last_mask'):
                    self.mask = self.last_mask.copy()
                    print("Undo complete")
            elif key == ord('+') or key == ord('='):  # Increase brush size
                self.brush_size = min(self.brush_size + 5, 100)
                print(f"Brush size: {self.brush_size}")
            elif key == ord('-'):  # Decrease brush size
                self.brush_size = max(self.brush_size - 5, 5)
                print(f"Brush size: {self.brush_size}")
            elif key == ord('s'):  # Save
                self.save_result()
        
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Replace 'your_image.jpg' with the path to your image
    image_path = "../../test-data/photos/approach7.jpg"  # Change this to your image file path
    
    try:
        editor = ImageBrushRemover(image_path)
        editor.run()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure the image file exists and the path is correct.")