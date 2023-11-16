import cv2
import os




class FaceCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.num_images = 0

    

    def capture_faces(self, olfolder_name):
        
        
        folder_name="data/"+olfolder_name
        os.makedirs(folder_name, exist_ok=True)
        while self.num_images < 20:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                face_filename = os.path.join(folder_name, f'face_{self.num_images}.jpg')
                cv2.imwrite(face_filename, face)
                self.num_images += 1
            
        return folder_name
        


