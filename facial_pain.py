import ctypes
import re
import cv2
import numpy as np
from keras.models import load_model
import os
import random
from playsound import playsound

suggestion_arr = os.listdir('./suggestion')


def detect_classify_display(frame, model_type='vgg16'):
    pred = 0
    emotions = ['pain', 'mild pain', 'pain',
                'no pain', 'pain', 'no pain', 'neutral']
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = frame_gray[y:y + h, x:x + w]
        if model_type == 'vgg16':
            face = cv2.resize(face, (197, 197)) / 255
            face = np.expand_dims(face, axis=0)
            
            face = np.expand_dims(face, axis=-1)
            face = np.repeat(face, 3, -1)
        else:
            face = cv2.resize(face, (48, 48)) / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
        predictions = model.predict(face)
        pred = np.argmax(predictions)
        prob = predictions[0, pred] * 100
        frame = cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
        cv2.putText(frame, emotions[pred] + f'   {prob:.2f}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.imshow('Emotion recognition', frame)
    return emotions[pred]


if __name__ == "__main__":

    # Loading model
    model_path = './models'
    model_type = re.search(r'(?<=fer2013_)(.*?)(?=_)', "fer2013_simple_CNN-e88-a0.65.hdf5")
    model = load_model("./models/saved_models/fer2013_simple_CNN-e88-a0.65.hdf5")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    count = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame -- Break!')
            break
        emotion = detect_classify_display(frame, model_type[0])
        if (emotion is "pain"):
            count = count + 1
        print(emotion, count)
        if count > 10:
            ctypes.windll.user32.MessageBoxW(0, "Pain Detected !", "Patient Status", 48 + 1)

            playsound('./audio/1.wav')
            r = int(random.random() * 12)
            break
        if cv2.waitKey(20) == 27:
            break
