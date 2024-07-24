import os
import cv2
from PIL import Image 
import numpy as np


path = os.path.dirname(os.path.abspath(__file__))
recognize = cv2.face.LBPHFaceRecognizer_create()
cascade_Path = path+r"\Classifiers\face.xml"
face_Cascade = cv2.CascadeClassifier(cascade_Path);
data_Path = path+r'\dataSet'

def get_img_label(datapath):
     image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
     images = []
     labels = []
     for image_path in image_paths:
         image_pil = Image.open(image_path).convert('L')
         image = np.array(image_pil, 'uint8')
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         print(nbr)
         faces = face_Cascade.detectMultiScale(image)
         for (x, y, W, H) in faces:
             images.append(image[y: y + H, x: x + W])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + H, x: x + W])
             cv2.waitKey(10)
     return images, labels


images, labels = get_img_label(data_Path)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognize.train(images, np.array(labels))
recognize.save(path+r'\trainer\trainer.yml')
cv2.destroyAllWindows()
