import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sklearn

# My file
import common

class DrivingLogLine:
    def __init__(self, line, img_dir_path):
        self.line = line
        #self.drivelogpath = drivelogpath
        self.img_dir_path = img_dir_path

    def get_img_path_from_column(self, column):
        source_path = self.line[column]
        filename = os.path.basename(source_path) #source_path.split('/')[-1]
        current_path = os.path.join(self.img_dir_path, filename)
        return current_path


    @property
    def center_img_path(self):
        return self.get_img_path_from_column(0)


    @property
    def angle(self):
        return float(self.line[3])


    @classmethod
    def read_lines_from_driving_log(cls, driving_log_path):
        lines = []
        with open(driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        # First line contains the name of the columns for me. If it just for some
        # library versions, then it is anyway ok to skip first frame.
        lines = lines[1:]

        img_dir_path = os.path.join(os.path.dirname(driving_log_path), 'IMG')
        lines = [cls(line, img_dir_path) for line in lines]

        return lines


def generator_from_lines(lines, augment, batch_size=32):
    num_samples = len(lines)
    while True:
        lines = sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            X,y = zip(*[make_sample_from_line(line, augment) for line in
                batch_lines])
            yield np.array(X),np.array(y)


crop_top = 70
crop_bottom = 25

def make_sample_from_line(line, augment):
    img = cv2.imread(line.center_img_path)
    img = common.preprocess_img(img)
    angle = line.angle

    if augment:
        if np.random.choice([False, True]):
            img = cv2.flip(img,1)
            angle = -angle
    
    return (img,angle)


def read_all_driving_logs(driving_log_paths):
    return list(sum(
        (DrivingLogLine.read_lines_from_driving_log(path)
            for path in driving_log_paths)
        , []))


driving_log_paths = [
    'udacity-my-driving-data/driving_log.csv',
    'drive2/driving_log.csv',
    'drive3/driving_log.csv',
    'recoverydriving1/recovery_log.csv',
    ]

lines = read_all_driving_logs(driving_log_paths)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

print('num frames: {}'.format(len(lines)))

train_generator = generator_from_lines(train_lines, True)
validation_generator = generator_from_lines(validation_lines, False)

from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Flatten, Lambda 
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
model.add(Cropping2D(cropping=((69,25),(0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='elu', name='conv1'))
model.add(Dropout(0.3))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='elu', name='conv2'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='elu', name='conv3'))
model.add(Conv2D(64, 3, 3, activation='elu', name='conv4'))
model.add(Conv2D(64, 3, 3, activation='elu', name='conv5'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1164, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(epsilon=1e-3))

model.fit_generator(
        train_generator, samples_per_epoch=2*len(train_lines),
        validation_data=validation_generator,
        nb_val_samples=len(validation_lines), nb_epoch=3)

model.save('model.h5')
