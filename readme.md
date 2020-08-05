# yolov4 face
model **[yolov4-416-face](https://drive.google.com/drive/folders/14jkY84nEdJeJjGypyTrrvqCH2CpEL_z-?usp=sharing)** train from yolov4 for custom face detector using wider-face data set 

you can use label_images.py for convert data from wider-face to darknet style for training (just change some directory depending on your project)
 
i convert my trained darknet model to keras by using [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite).
note that you might want to change the obj.names file to your custom names file
