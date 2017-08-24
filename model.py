import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sklearn

def read_lines_from_driving_log(path):
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    # First line contains the name of the columns for me. If it just for some
    # library versions, then it is anyway ok to skip first frame.
    lines = lines[1:]

    return lines


def generator_from_lines(lines, batch_size=32):
    num_samples = len(lines)
    while True:
        lines = sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            yield lines_to_X_and_y(batch_lines)


# Convert a set of lines from a driving log to datas and expected values.
def lines_to_X_and_y(batch_lines):
    images = []
    measurements = []
    for line in batch_lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = os.path.join('udacity-my-driving-data', 'IMG', filename)
        image = cv2.imread(current_path)
        angle = float(line[3])
        images.append(image)
        measurements.append(angle)

        # Add vertical flip
        images.append(cv2.flip(image,1))
        measurements.append(-angle)
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train


lines = read_lines_from_driving_log('udacity-my-driving-data/driving_log.csv')
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

print('num frames: {}'.format(len(lines)))

train_generator = generator_from_lines(train_lines)
validation_generator = generator_from_lines(validation_lines)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(
        train_generator, samples_per_epoch=2*len(train_lines),
        validation_data=validation_generator,
        nb_val_samples=len(validation_lines), nb_epoch=7)

model.save('model.h5')
