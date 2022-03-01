import cv2
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras import backend
backend.set_image_data_format('channels_last')
# backend.set_image_dim_ordering('th')

#Map letters and prediction
alphabets = 'abcdefghijklmnopqrstuvwxyz'
mapping_letter = {}
for i,l in enumerate(alphabets):
    mapping_letter[l] = i
mapping_letter = {v:k for k,v in mapping_letter.items()}

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

model = keras.models.load_model(r"\model\sign_alphabets.h5")
# model.summary()


class OpenCam(object):

    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)
        self.num_frames = 0

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frames(self):

        ret, frame = self.video.read()

        # flipping the frame to prevent inverted image of captured

        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        # segmenting the hand region
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        x = image.img_to_array(img)
        #print("before",x.shape)
        images = np.expand_dims(x, axis=0)
        #print(images.shape)

        # Drawing contours around hand segment
        cv2.imshow("Thresholded Hand Image", img)
        pred = model.predict(images)
        pred_num = int(np.argmax(pred, axis=1))
        pred_text = mapping_letter[pred_num]
        cv2.putText(frame_copy, str(pred_text), (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
                                                        ROI_bottom), (255, 128, 0), 3)

        # incrementing the number of frames for tracking
        self.num_frames += 1

        # Display the frame with segmented hand
        cv2.putText(frame_copy, "Hand sign recognition_ _ _",
                    (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
        cv2.imshow("Sign Detection", frame_copy)

        # Close windows with Esc
        # k = cv2.waitKey(5) & 0xFF

        ret, jpeg = cv2.imencode('.jpg', frame_copy)
        return jpeg.tobytes()

        #if k == 27:
            #break

    # Release the camera and destroy all the windows
    # cam.release()
    # cv2.destroyAllWindows()
