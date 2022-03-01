# Flask libraries
from flask import Flask, render_template, Response, request
import cv2
import os, sys
import numpy as np
from threading import Thread
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS

# OpenCV Libraries
from opencv import OpenCam

# import cv2
# import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras import backend
backend.set_image_data_format('channels_last')
# backend.set_image_dim_ordering('th')

global switch
switch=1
#
# #Map letters and prediction
# alphabets = 'abcdefghijklmnopqrstuvwxyz'
# mapping_letter = {}
# for i,l in enumerate(alphabets):
#     mapping_letter[l] = i
# mapping_letter = {v:k for k,v in mapping_letter.items()}
#
# ROI_top = 100
# ROI_bottom = 300
# ROI_right = 150
# ROI_left = 350

# model = keras.models.load_model(r"..\team-20\model\sign_alphabets.h5")
# model.summary()

#instatiate flask app
#app = Flask(__name__, template_folder='.\templates')

#Save model
# model_path = r"..\team-20\model\sign_alphabets.h5"
#
# # Load your trained model
# model = load_model(model_path)
# model._make_predict_function()

#print('OpenCv Working')

#camera_port = 0
#cam = cv2.VideoCapture(camera_port)
#num_frames = 0

app = Flask(__name__)

#camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alphabets')
def alphabets():
    return render_template('alphabets.html')

@app.route('/numbers')
def numbers():
    return render_template('numbers.html')

def gen_frames(camera):
    while True:
        #get camera frame
        if switch == 0:
            break
        frame = camera.get_frames()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(OpenCam()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(OpenCam()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames(OpenCam):
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-cd Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# Route for Button
@app.route('/start_button',methods=['POST','GET'])
def start_button():
    global switch
    #global switch,camera
    if request.method == 'POST':
        if(switch==1):
            switch=0
        else:
            switch = 1
            video_feed()
        # if  request.form.get('stop') == 'Start':
        #     if(switch==1):
        #         switch=0
        #         #camera.release()
        #         #cv2.destroyAllWindows()
        #     else:
        #         switch = 1
        #         video_feed()
                #camera = cv2.VideoCapture(0)

    elif request.method=='GET':
        return render_template('alphabets.html')

    return render_template('alphabets.html')

if __name__ == "__main__":
    app.run(debug=False)
    #app.run()

# Release the camera and destroy all the windows
# cam.release()
cv2.destroyAllWindows()
