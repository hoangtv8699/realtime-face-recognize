import numpy as np
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load faces
    data = np.load('../videos/5-celebrity-faces-dataset.npz')
    testX_faces = data['arr_2']
    # load dataset
    dataset = np.load('../videos/5-celebrity-faces-embedding.npz')
    trainX, trainy, testX, testy = dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3']
    # normalize input
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode target
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # test model on random example from test dataset
    selection = random.choice([i for i in range(testX.shape[0])])
    random_face_pixels = testX_faces[selection]
    random_face_emb = testX[selection]
    random_face_class = testy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # predict for the face
    samples = np.expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_prob = yhat_prob[0, class_index]*100
    predict_name = out_encoder.inverse_transform(yhat_class)
    print('predicted: {} {}'.format(predict_name[0], class_prob))
    print('expected: {}'.format(random_face_name[0]))
    # plot for fun
    plt.imshow(random_face_pixels)
    title = '{} {}'.format(predict_name[0], class_prob)
    plt.title(title)
    plt.show()