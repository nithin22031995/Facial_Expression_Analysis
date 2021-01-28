#Last login: Thu Jan 21 11:14:51 on ttys002
#(base) nithinvenkat@student-10-201-23-058 ~ % ipython
#Python 3.7.6 (default, Jan  8 2020, 13:42:34)
#Type 'copyright', 'credits' or 'license' for more information
#IPython 7.12.0 -- An enhanced Interactive Python. Type '?' for help.


import sys
import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from utility.datasets import get_labels
from utility.inference import detect_faces
from utility.inference import draw_text
from utility.inference import draw_bounding_box
from utility.inference import apply_offsets
from utility.inference import load_detection_model
from utility.inference import load_image
from utility.preprocessor import preprocess_input


#Loading the data and pre-trained models

image_path = '../Images/blur_image.jpeg'
detection_model_path = '../pretrained_models/face_detection/haarcascade_frontalface_default.xml'
emotion_model_path = '../pretrained_models/facial_expression/simpleCNN.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_TRIPLEX


#Parameters for shape of the Bounding-Box

emotion_offsets = (20, 40)
emotion_offsets = (0, 0)


#Loading the pre-trained models and using model shapes for the inference

face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]


#Reading the images and converting to appropriate color codes for processing
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

#Detecting the face using Haar Cascade model

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    #print(face_coordinates)
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]

    if emotion_text == emotion_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)
	
print('Bounding-Box and Facial Experssion Generated')

#Facial Expression (Sentiment Analysis) on a Face, saving on local folder

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../samples/facial_expression_prediction.jpeg', bgr_image)

