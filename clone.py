import csv
import cv2
import numpy as np
import socketio
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Cropping2D

lines = []
with open('data\\driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
                lines.append(line)
images = []
measurements = []
for line in lines:
        source_path = line[0]
        filename = source_path.split('\\')[-1]
        current_path = 'data\\IMG\\' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# set up lambda layer
model = Sequential()
model.add(Cropping2D(cropping=((65, 25), (0, 0)), input_shape=np.shape(images[0])))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(8, 3, 3, activation='relu'))
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
print("Model saved")

