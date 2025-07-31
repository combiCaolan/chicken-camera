import cv2
import numpy as np

# Load image
image = cv2.imread("image.jpg")
clone = image.copy()

# Mask for the painted area (1-channel grayscale)
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Brush state
drawing = False
brush_size = 20

# Mouse callback to paint over image
def paint(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Setup window and callback
cv2.namedWindow("Select ROI - Paint to Keep Area")
cv2.setMouseCallback("Select ROI - Paint to Keep Area", paint)

while True:
    # Overlay painted mask onto the original image for preview
    overlay = clone.copy()
    overlay[mask == 255] = [0, 255, 0]  # Show painted area in green

    cv2.imshow("Select ROI - Paint to Keep Area", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # Quit without saving
        break
    elif key == ord("s"):  # Save painted region and quit
        # Apply the mask to the image
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite("selected_area_only.jpg", result)
        break

cv2.destroyAllWindows()
