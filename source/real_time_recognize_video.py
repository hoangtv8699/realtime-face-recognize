import colorsys

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from tensorflow import keras
import cv2
import random
import time
from source.detect_face import draw_bbox
from source.embedding_face import extract_faces, get_embedding


if __name__ == '__main__':
    # load dataset
    dataset = np.load('../5-celebrity-faces-dataset/5-celebrity-faces-embedding.npz')
    trainX, trainy, testX, testy = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3']
    # normalize input
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    # label encode target
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)

    class_name = out_encoder.classes_
    trainy = out_encoder.transform(trainy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)

    input_size = 416
    image_path = '../image/bao_ngu.jpg'
    model_path = '../models/keras_model/yolov4-face-tiny-416'
    # iou score thresshold
    iou = 0.45
    score = 0.25
    threshold = 0.8
    # load model yolov4-face
    saved_model_loaded = tf.keras.models.load_model(model_path, compile=False)
    infer = saved_model_loaded.signatures['serving_default']

    # load model facenet
    facenet = tf.keras.models.load_model('../models/keras_model/facenet_keras.h5', compile=False)

    ben_video = cv2.VideoCapture('../videos/ben_afleck/ben afleck.mp4')
    ret, frame = ben_video.read()

    while ret:
        seconds = time.time()
        boxes, faces_array = extract_faces(frame, infer)
        # # embedding face
        # embedded_faces = []
        # for i in range(len(faces_array)):
        #     embedded_face = faces_array[i]
        #     embedded_face = get_embedding(facenet, embedded_face)
        #     embedded_faces.append(embedded_face)
        #
        # # predict face
        # yhat_proba = model.predict_proba(embedded_faces)
        # predict_names = []
        # for proba in yhat_proba:
        #     max_proba = np.argmax(proba)
        #     if proba[max_proba] > threshold:
        #         predict_names.append(class_name[max_proba])
        #     else:
        #         predict_names.append("Unknown")

        for i in range(len(boxes)):
            # get box
            x1, y1, x2, y2 = boxes[i]
            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # set name
            # label = "{}".format(predict_names[i])
            label = "face"
            # input name of face
            cv2.putText(frame, label, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
        print(time.time() - seconds)
        cv2.imshow('frame', frame)
        ret, frame = ben_video.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
