# import os
# import random
import pdb
import warnings
warnings.filterwarnings("ignore")
import h5py

import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras import optimizers

from keras.models import load_model
from res_attn_net_keras import ResidualAttentionNetwork

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

batch_size = 12

epochs = 10

num_classes = 2

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=.33
)

train_generator = train_datagen.flow_from_directory(
    directory="../cats_and_dogs_keras/trn/",
    shuffle=True,
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=batch_size,
    subset='training'
)

valid_generator = train_datagen.flow_from_directory(
    directory="../cats_and_dogs_keras/val/",
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode="categorical",
    color_mode='grayscale',
    shuffle=True,
    subset='validation'
)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

model_path = "cvd-model.h5"
# early_stop = EarlyStopping(monitor='val_acc',  verbose=1, patience=50)
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                             save_best_only=True)
csv_logger = CSVLogger("cvd-model-history.csv", append=True)

callbacks = [checkpoint, csv_logger]

# Model Training
with tf.device('/gpu:0'):
    # model = ResidualAttentionNetwork(
    #             input_shape=IMAGE_SHAPE,
    #             n_classes=num_classes,
    #             activation='softmax').build_model()
    model = load_model('cats-vs-dogs-model.h5')
    pdb.set_trace()

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=epochs,
                                  use_multiprocessing=True, workers=40)
    model.save('cvd_main.h5')
