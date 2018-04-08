import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
import sklearn
from sklearn.model_selection import train_test_split

samples = []
# Extract the list of all samples, this will be fed to the generator to process it in batches.
with open('data2\\driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    sklearn.utils.shuffle(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            # Extract the images and steering angles
            for batch_sample in batch_samples:
                image_center = cv2.imread(batch_sample[0])
                image_left = cv2.imread(batch_sample[1])
                image_right = cv2.imread(batch_sample[2])
                images.extend([image_center, image_left, image_right])
                steering = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.20
                steering_left = steering + correction
                steering_right = steering - correction

                # add images and angles to data set
                steerings.extend([steering, steering_left, steering_right])

            # trim image to only see section with road
            X_train_gen = np.array(images)
            y_train_gen = np.array(steerings)
            yield sklearn.utils.shuffle(X_train_gen, y_train_gen)


# compile and train the model using the generator function
BATCH_SIZE = 64
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = Sequential()
top_crop = 65
bottom_crop = 25

ch, row, col = 3, 160 - (top_crop + bottom_crop), 320  # Trimmed image format

# Pre-process incoming data, centered around zero with small standard deviation

model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0, 0)), input_shape=[160, 320, 3]))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=[row, col, ch]))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())

# Fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, validation_data=validation_generator,
                    validation_steps=len(validation_samples)/BATCH_SIZE, epochs=3)
model.save('model.h5')
print("Model saved")
