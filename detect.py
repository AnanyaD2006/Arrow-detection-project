import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("arrow_classifier_keras_gray.keras")


class_labels = ['down', 'left', 'right', 'up']  
img_size = 60

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    adaptive_thresh = cv2.adaptiveThreshold(
    blur, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 
    11, 2)

    
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 500:  
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (img_size, img_size))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = roi_normalized.reshape(1, img_size, img_size, 1)
    

    prediction = model.predict(roi_reshaped)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = f"{class_labels[class_index]} ({confidence:.2f})"

    
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Arrow Classification", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()










