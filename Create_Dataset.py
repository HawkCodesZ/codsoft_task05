import os
import cv2

base_dir = os.path.dirname(os.path.abspath(__file__))
video_capture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(os.path.join(base_dir, 'Classifiers', 'face.xml'))
img_counter = 1
offset = 50
user_id = input('Enter your ID: ')

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in detected_faces:
        img_path = os.path.join('dataSet', f'face-{user_id}.{img_counter}.jpg')
        cv2.imwrite(img_path, gray_frame[y-offset:y+h+offset, x-offset:x+w+offset])
        cv2.rectangle(frame, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)
        cv2.imshow('Captured Face', frame[y-offset:y+h+offset, x-offset:x+w+offset])
        cv2.waitKey(100)
        img_counter += 1
    
    if img_counter > 20:
        video_capture.release()
        cv2.destroyAllWindows()
        break

