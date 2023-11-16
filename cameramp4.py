import cv2


import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))


# Define a class to handle the video stream
class VideoStream(object):
    def __init__(self, file_path):
        self.video = cv2.VideoCapture(file_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        print(self.video)
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                
                  break

            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
            if frame is not None and isinstance(frame, np.ndarray):
                for x, y, w, h in faces:
                      
                    img = rgb_img[y:y+h, x:x+w]
                    img = cv2.resize(img, (160, 160))
                    img = np.expand_dims(img, axis=0)
                    ypred = facenet.embeddings(img)
                    face_name = model.predict(ypred)
                    final_name = encoder.inverse_transform(face_name)[0]
                    modified_frame = frame.copy()  # Create a copy of the original frame
                    cv2.rectangle(modified_frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
                    cv2.putText(modified_frame, str(final_name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

                    ret, buffer = cv2.imencode('.jpg', modified_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                    print("Invalid frame")        


