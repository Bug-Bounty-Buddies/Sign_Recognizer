import cv2

# Set the correct index for DroidCam, e.g., 1 instead of 0
cap = cv2.VideoCapture(1)  # Change 1 to whatever index corresponds to DroidCam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('DroidCam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
