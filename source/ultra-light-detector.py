import cv2
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from onnx_tf.backend import prepare
import time

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

from source.embedding_face import get_embedding
import tensorflow as tf


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


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
    # input size
    ultra_light_input = (640, 480)
    facenet_input = (160, 160)
    # face recognize threshold
    threshold = 0.9

    # load model
    onnx_model = onnx.load('../models/ultra_light_640.onnx')
    predictor = prepare(onnx_model)

    ort_session = ort.InferenceSession("../models/ultra_light_640.onnx")
    input_name = ort_session.get_inputs()[0].name

    # load model facenet
    facenet = tf.keras.models.load_model('../models/keras_model/facenet_keras.h5', compile=False)

    # read frame
    ben_video = cv2.VideoCapture('../videos/ben_afleck/ben afleck.mp4')
    ret, frame = ben_video.read()

    while ret:
        seconds = time.time()
        # get original size
        image_h, image_w, channels = frame.shape
        # change to RGB and resize
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, ultra_light_input)
        # pipeline
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128

        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)

        image = image.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: image})

        boxes, labels, probs = predict(image_w, image_h, confidences, boxes, 0.7)

        faces_array = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = abs(boxes[i][0]), abs(boxes[i][1]), abs(boxes[i][2]), abs(boxes[i][3])
            face = frame[y1:y2, x1:x2, ]
            face_array = Image.fromarray(face)
            face_array = face_array.resize(facenet_input)
            face_array = np.asarray(face_array)
            faces_array.append(face_array)

        embedded_faces = []
        for i in range(len(faces_array)):
            embedded_face = faces_array[i]
            embedded_face = get_embedding(facenet, embedded_face)
            embedded_faces.append(embedded_face)

        # predict face
        yhat_proba = model.predict_proba(embedded_faces)
        predict_names = []
        for proba in yhat_proba:
            max_proba = np.argmax(proba)
            if proba[max_proba] > threshold:
                predict_names.append(class_name[max_proba])
            else:
                predict_names.append("Unknown")

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
            cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            text = f"{predict_names[i]}"
            cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        print(time.time() - seconds)
        ret, frame = ben_video.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
