import cv2
import numpy as np

video_path = "../../test-data/videos/REALISTIC-RECORDING.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_chicken = np.array([90, 50, 80])
    upper_chicken = np.array([110, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_chicken, upper_chicken)

    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chicken_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Adjust this as needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            chicken_count += 1

    cv2.putText(frame, f'Chickens: {chicken_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.waitKey(100)
    cv2.imshow("Chicken Detection", frame)
    cv2.imshow("Chicken Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()