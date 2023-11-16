import cv2


import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

facenet = FaceNet()



class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
        self.Y = self.faces_embeddings['arr_1']
        self.encoder = LabelEncoder()
        self.encoder.fit(self.Y)
        self.haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        ret,frame = self.video.read()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x,y,w,h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv2.resize(img, (160,160)) # 1x160x160x3
            img = np.expand_dims(img,axis=0)
            ypred = facenet.embeddings(img)
            face_name = self.model.predict(ypred)
            final_name = self.encoder.inverse_transform(face_name)[0]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
            cv2.putText(frame, str(final_name), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,0,255), 3, cv2.LINE_AA)

        #cv2.imshow("Face Recognition:", frame)




        ret,jpg = cv2.imencode('.jpg',frame)
        return jpg.tobytes()         