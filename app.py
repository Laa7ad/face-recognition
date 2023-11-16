from flask  import Flask,render_template,redirect,url_for,Request,Response,request,send_from_directory,jsonify

from cameramp4 import VideoStream
import cv2
from addpersonne import FaceCapture
from searchsengine import ReverseImageSearch 
from _curses import *
from googleapiclient.errors import HttpError
import os
from camera import Video
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import time
from model import FACELOADING
from model import get_embedding
from flask_socketio import SocketIO
import json

app = Flask(__name__)

@app.route('/data/<path:filename>')
def data(filename):
    return send_from_directory('data', filename)

def check_camera():
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        cap.release()
        return True
    else:
        return False
@app.route('/train_model', methods=['GET','POST'])

def train_model():
    try:
                embedder = FaceNet()
                directory_path = 'data'
                faceloading = FACELOADING(directory_path)
                X, Y = faceloading.load_classes()  
                #detector = MTCNN()
                EMBEDDED_X = []

                for face_img in X:
                    embedding = get_embedding(embedder, face_img)
                    EMBEDDED_X.append(embedding)

                EMBEDDED_X = np.asarray(EMBEDDED_X)


                np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)

                encoder = LabelEncoder()
                encoder.fit(Y)
                Y = encoder.transform(Y)

                X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

                model = SVC(kernel='linear', probability=True)
                model.fit(X_train, Y_train)

                ypreds_train = model.predict(X_train)
                ypreds_test = model.predict(X_test)

                acc_train= accuracy_score(Y_train, ypreds_train)
                acc_test =accuracy_score(Y_test,ypreds_test)



                with open('svm_model_160x160.pkl','wb') as f:
                    pickle.dump(model,f)

                response = {
                    'status': 'success',
                    'message': 'Model trained successfully!',
                    'accuracy_train': acc_train,
                    'accuracy_test': acc_test
                }

               
                return Response(f"data: {json.dumps(response)}\n\n", content_type='text/event-stream')

    except Exception as e:
            # If an error occurs during training, handle the exception and send an error response
            error_response = {
                'status': 'error',
                'message': f'Error occurred during training: {str(e)}'
            }
            return Response(f"data: {json.dumps(response)}\n\n", content_type='text/event-stream')

@app.route('/',methods=['GET','POST'])
def index():
    if(check_camera()):
          return render_template('index.html')
    else:
          return render_template('no_camera.html')      
def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')

def video():
    
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/film')
def film():
    return render_template('film.html')


@app.route("/search",methods=["POST", "GET"])
def search():
	directory_path = 'data'
	# List all files (including in subdirectories)
	all_files = [os.path.join(root, file) for root, dirs, files in os.walk(directory_path) for file in files]

	# Filter only the files (excluding directories)
	file_list = [file for file in all_files if os.path.isfile(file)]
	if request.method == "POST":
		result = request.form
		selected = result['image_select']
		try:
			api_key = 'AIzaSyC3Bz_Q5e93BSgg-f1uZuvraBaT-gdpToY'
			search_engine_id = 'a44a1091feb1d4cde'
			
			image_search = ReverseImageSearch(api_key, search_engine_id)
			image_path = selected
			results_images = image_search.search_by_image(image_path)
			results_image_url =image_search.get_first_result_url(results_images)
		except HttpError as e:
				error_details = e._get_reason()
				print(f"HTTP Error: {error_details}")
				results_image_url =  "Not found  by google search images"
		 
		return render_template("searchimage.html",images=file_list,simage=selected,rsimages= results_image_url)

	else:
	    return render_template("searchimage.html",images=file_list)


@app.route('/stream')

def stream(path):
    frame = VideoStream(path)
    return Response(frame.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/load_function', methods=['POST'])
def load_function():
    selected_option = request.form['option']
    
    if selected_option == 'none':
        return render_template('no_camera.html')   
    elif selected_option == 'film':
        path = "mp4\mp4video14.mp4"
        return stream(path)
    elif selected_option == 'film2':
        path ="mp4/Robert Downey Jr.mp4"
        return stream(path)
    
@app.route('/capture',methods=['POST','GET'])

def capture():
    if request.method == 'POST':
        face_capture = FaceCapture()
        folder_name = request.form.get('folder_name')
        result = face_capture.capture_faces(folder_name)
        return result
    if(check_camera()):
         reslt = True
    else:
          reslt = False    
    return render_template('capture.html',images=reslt )
    

if __name__ == "__main__":
    app.run(debug=True)


