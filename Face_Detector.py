import cv2
import os
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(os.path.join(base_dir, 'trainer', 'trainer.yml'))
cascade_file = os.path.join(base_dir, 'Classifiers', 'face.xml')
face_detector = cv2.CascadeClassifier(cascade_file)

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX  

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, W, H) in faces:
        predicted_id, confidence = face_recognizer.predict(gray_frame[y:y+H, x:x+W])
        cv2.rectangle(frame, (x-50, y-50), (x+W+50, y+H+50), (225, 0, 0), 2)
        if predicted_id == 1:
            predicted_id = "Sagnik"
        label = f"ID: {predicted_id}, Conf: {confidence:.2f}"
        cv2.putText(frame, label, (x, y+H), font, 1.1, (0, 255, 0), 2)  

    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()







