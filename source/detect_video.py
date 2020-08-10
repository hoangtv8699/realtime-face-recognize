import colorsys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import cv2
import random
import time
from source.detect_face import draw_bbox


if __name__ == '__main__':
    input_size = 416
    image_path = '../image/bao_ngu.jpg'
    model_path = '../models/keras_model/yolov4-416-face'

    iou = 0.45
    score = 0.25

    saved_model_loaded = tf.keras.models.load_model(model_path)
    infer = saved_model_loaded.signatures['serving_default']

    ben_video = cv2.VideoCapture('../videos/ben_afleck/ben afleck.mp4')
    ret, frame = ben_video.read()

    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        frame = draw_bbox(frame, pred_bbox)
        cv2.imshow('frame', frame)
        ret, frame = ben_video.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
