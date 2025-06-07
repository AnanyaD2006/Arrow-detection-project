import cv2
import numpy as np
import os
from random import shuffle
import glob
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping




train_dir = "D:\\VSC\\train"
test_dir = "D:\\VSC\\test"
MODEL_NAME = "arrow_classifier_keras_gray.h5"
img_size = 60


classifier = Sequential()
classifier.add(Input(shape=(img_size, img_size, 1)))
classifier.add(Conv2D(32, (3, 3),padding='same'))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(64, (3, 3), padding='same'))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(128))
classifier.add(BatchNormalization())
classifier.add(Activation("relu"))
classifier.add(Dropout(0.3))

classifier.add(Dense(4))  
classifier.add(BatchNormalization())
classifier.add(Activation("softmax"))


classifier.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1./255, 
    rotation_range=25,           \
    width_shift_range=0.2,      
    height_shift_range=0.2,      
    shear_range=0.2,             
    zoom_range=0.2,              
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255 )

training_set = train_datagen.flow_from_directory(
    'D:\\VSC\\train',
    color_mode="grayscale",
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)



test_set = test_datagen.flow_from_directory(
    'D:\\VSC\\test',
    color_mode="grayscale",
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


with open("class_indices.txt", "w",encoding="utf-8") as indices_file:
    classifier.summary(print_fn=lambda x: indices_file.write(x + "\n"))
    indices_file.write("\nTraining set class indices:\n" + str(training_set.class_indices))
    indices_file.write("\nTest set class indices:\n" + str(test_set.class_indices))

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss"
)


tbCallBack = TensorBoard(log_dir='./log', write_graph=True)
classifier.fit(
    training_set,
    steps_per_epoch=200,
    epochs=15,
    validation_data=test_set,
    validation_steps=50,
    shuffle=True,
    callbacks=[early_stop,checkpoint_cb]
)


classifier.save("arrow_classifier_keras_gray.keras")
