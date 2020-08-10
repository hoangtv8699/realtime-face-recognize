from PIL import Image
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    image = Image.open('../image/bao_ngu.jpg')
    image = image.resize((416, 416))
    pixel = np.asarray(image)
    pixel = pixel[np.newaxis, ...]
    print(pixel.shape)

    model = tf.keras.models.load_model("../models/keras model/yolov4-416-face", custom_objects={'tf': tf})
    outs = model.predict(pixel)
    for out in outs:
        print(out.shape)
