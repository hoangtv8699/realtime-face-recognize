from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sklearn
import cv2

input_size = 416
image_path = '../image/bao_ngu.jpg'
model_path = '../models/keras_model/yolov4-416-face'

iou = 0.45
score = 0.25

saved_model_loaded = tf.keras.models.load_model(model_path)
infer = saved_model_loaded.signatures['serving_default']


# extract a single face from given photo
def extract_face(filename, require_size=(160, 160)):
    # load image from file name
    original_image = cv2.imread(filename)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # save original image shape
    image_h, image_w, image_channels = original_image.shape

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = [image_data]
    images_data = np.asarray(images_data).astype(np.float32)
    # detect face
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
    box = boxes[0][0]
    y1 = int(box[0] * image_h)
    y2 = int(box[2] * image_h)
    x1 = int(box[1] * image_w)
    x2 = int(box[3] * image_w)

    # extract face
    original_image = original_image[y1:y2, x1:x2, :]
    # resize image
    image = Image.fromarray(original_image)
    image = image.resize(require_size)
    face_array = np.asarray(image)
    return face_array


# def extract_faces(frame, require_size=(160, 160)):
#     pixels = np.asarray(frame)
#     # create detector, using default weights
#     detector = MTCNN()
#     # detect face in image
#     results = detector.detect_faces(pixels)
#     # get bounding box from first face
#     boxes = list()
#     faces_array = list()
#     for i in range(len(results)):
#         x1, y1, width, height = results[i]['box']
#         # get abs
#         x1, y1 = abs(x1), abs(y1)
#         x2, y2 = x1 + width, y1 + height
#         # extract face
#         face = pixels[y1:y2, x1:x2]
#         # resize image
#         image = Image.fromarray(face)
#         image = image.resize(require_size)
#         face_array = np.asarray(image)
#         # store
#         boxes.append([x1, y1, x2, y2])
#         faces_array.append(faces_array)
#     return boxes, faces_array


# load images and extract faces for all images in directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in os.listdir(directory):
        # get path file
        path = directory + filename
        # extract face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    # x for faces and y for labels
    x, y = list(), list()
    # enumerate directory
    for subdir in os.listdir(directory):
        # get path subdir
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the path
        faces = load_faces(path)
        # create label
        labels = [subdir for _ in range(len(faces))]
        # summarize
        print('loaded {} examples for class: {}'.format(len(faces), subdir))
        # store
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x), np.asarray(y)


def get_embedding(model, face_pixels):
    # scales pixels values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face to one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make predict to get embedding
    yhat = model.predict(samples)
    return yhat[0]


if __name__ == '__main__':
    # load train dataset
    trainX, trainy = load_dataset('../5-celebrity-faces-dataset/train/')
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = load_dataset('../5-celebrity-faces-dataset/val/')
    # load model
    model = tf.keras.models.load_model('../models/keras_model/facenet_keras.h5')
    print('loaded model')
    # save face dataset
    np.savez_compressed('../videos/5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)
    # convert each face in train set to embedding
    newTrain = list()
    for face_pixel in trainX:
        embedding = get_embedding(model, face_pixel)
        newTrain.append(embedding)
    newTrain = np.asarray(newTrain)
    print(newTrain.shape)
    # convert each image in test set to embedding
    newTest = list()
    for face_pixel in testX:
        embedding = get_embedding(model, face_pixel)
        newTest.append(embedding)
    newTest = np.asarray(newTest)
    print(newTest.shape)
    # save arrays to one file in compressed format
    np.savez_compressed('../videos/5-celebrity-faces-embedding.npz', newTrain, trainy, newTest, testy)
