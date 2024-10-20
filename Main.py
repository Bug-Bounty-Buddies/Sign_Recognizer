import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained models with different accuracies
model1 = load_model('traffic_classifier.h5')  # First model with lower accuracy
model2 = load_model('best_model.h5')  # Second model with higher accuracy

# Dictionary to label all traffic sign classes
classes = {
    1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)', 5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)', 8: 'Speed limit (100km/h)', 9: 'Speed limit (120km/h)',
    10: 'No passing', 11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection',
    13: 'Priority road', 14: 'Yield', 15: 'Stop', 16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited', 18: 'No entry', 19: 'General caution',
    20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve',
    23: 'Bumpy road', 24: 'Slippery road', 25: 'Road narrows on the right',
    26: 'Road work', 27: 'Traffic signals', 28: 'Pedestrians', 29: 'Children crossing',
    30: 'Bicycles crossing', 31: 'Beware of ice/snow', 32: 'Wild animals crossing',
    33: 'End speed + passing limits', 34: 'Turn right ahead', 35: 'Turn left ahead',
    36: 'Ahead only', 37: 'Go straight or right', 38: 'Go straight or left',
    39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory',
    42: 'End of no passing', 43: 'End no passing veh > 3.5 tons'
}

# Preprocess the image for both models
def preprocess_image(img):
    img = cv2.resize(img, (30, 30))  # Resize for model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img

# Function to detect sign and use both models
def detect_sign(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this value based on your needs
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]

            # Preprocess the ROI
            preprocessed = preprocess_image(roi)

            # Make predictions with both models
            prediction1 = model1.predict(preprocessed)  # First model prediction
            prediction2 = model2.predict(preprocessed)  # Second model prediction

            # Get predicted class for both models
            sign_class1 = np.argmax(prediction1) + 1
            sign_class2 = np.argmax(prediction2) + 1

            # Display the traffic sign name from the model with higher confidence
            if sign_class2 != sign_class1:  # Use second model if it gives different result
                sign_class = sign_class2
            else:
                sign_class = sign_class1

            sign_name = classes[sign_class]

            # Draw rectangle and label with only the sign name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, sign_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Main function to run the program
def main():
    cap = cv2.VideoCapture(1)  # Change this to the index you found for DroidCam
    cap.set(3, 640)  # Set the width of the frame
    cap.set(4, 480)  # Set the height of the frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect and recognize traffic signs using both models
        processed_frame = detect_sign(frame)

        # Display the result
        cv2.imshow('Traffic Sign Detection', processed_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
