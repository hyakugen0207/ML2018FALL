import pandas as pd
import numpy as np
import keras
import sys
import os
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers

train_data_path = sys.argv[1]

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1)

# setting
validate = 1
epochs = 20000
#convert feature to numpy ndarray
train_data = pd.read_csv(train_data_path, encoding='big5', dtype='O')
split_feature = train_data['feature'].str.split(' ')
train_feature = np.zeros(((len(train_data)//10)*(10-validate),48,48,1), dtype='float32')
validation_feature = np.zeros((len(train_data)-len(train_feature),48,48,1), dtype='float32')
log = 0
for index in range(0,len(split_feature)):
    if index < (len(split_feature)//10)*(10-validate):
        c = np.asarray(split_feature[index], dtype=float)
        train_feature[index] = c.reshape(48,48,1)
        log = index
    else:
        c = np.asarray(split_feature[index], dtype=float)
        validation_feature[index-log-1] = c.reshape(48,48,1)

t_label = np.asarray(train_data['label'], dtype=float)
one_hot_label = encode(t_label)
train_label = one_hot_label[0:(len(train_data)//10)*(10-validate)]
validation_label = one_hot_label[(len(train_data)//10)*(10-validate):]

model_chk_path2 = 'model_train'
model_chk_path = model_chk_path2 + '.h5'
early_stopping = EarlyStopping(monitor='val_acc', patience= 200, verbose=1, mode='max')
saveBestModel = ModelCheckpoint(model_chk_path, monitor="val_acc", save_best_only=True, save_weights_only=False)

model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D((2, 2))) #23*23*128
model.add(Dropout(0.5))
model.add(Conv2D(128,kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2))) #10*10*64
model.add(Dropout(0.4))
model.add(Conv2D(256,kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(512,kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(keras.layers.Flatten())
model.add(Dense(175))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#train_history = model.fit(x=train_feature,  y=one_hot_label, validation_split=0.1,  epochs=2000, batch_size=1000, verbose=1, callbacks=[early_stopping, saveBestModel])
datagen.fit(train_feature)
train_history = model.fit_generator(datagen.flow(train_feature, train_label, batch_size=32),
                    steps_per_epoch=len(train_feature) / 32, epochs=epochs, validation_data=(validation_feature, validation_label), verbose=1, callbacks=[early_stopping, saveBestModel])

model.save(model_chk_path)