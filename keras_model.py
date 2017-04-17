import pickle
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# load dataset
data_file = './np_dataset.p'
with open(data_file,mode='rb') as f:
    dataset = pickle.load(f)

X = dataset['images']
y = dataset['steering']

print('data is loaded')

new_height = 16
new_width = 64

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) -0.5, input_shape = (new_height,new_width,3)))
model.add(Convolution2D(1,1,1))
model.add(Convolution2D(16,3,3, activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(1,2)))
#model.add(Convolution2D(16,3,3,activation='relu'))
#model.add(Convolution2D(16,3,3,activation='relu'))
#model.add(MaxPooling2D(2,2))
model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X, y, shuffle = True, nb_epoch = 1)
model.summary()
model.save('model.h5')
