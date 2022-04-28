# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:18:56 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
from keras.preprocessing import image
import os
import cv2

import h5py
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.utils import multi_gpu_model

from keras.engine.topology import Layer

import keras
import keras.applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

imglist = pd.read_excel('F:/rice_heading/train_label.xls')
n_patch = imglist.shape[0]
Y = imglist.label.values
Y = Y.astype('float32')

input_image = np.zeros([n_patch,256,256,3],'float32')
for i in range(n_patch):
    img=image.load_img('F:/rice_heading/train_img/'+imglist.img_name[i])
    x = image.img_to_array(img)
    input_image[i]=x
    
X_train, X_val, Y_train, Y_val = train_test_split(input_image, Y, test_size=0.2, random_state=2)

trainDatagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.05,
        height_shift_range=0.05,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        #rescale = 1/255,
        #validation_split=0.2,
        #cval=0
        )

# data augmentation with multiple random transformation
trainGenerator = trainDatagen.flow(X_train, Y_train, batch_size=128, shuffle=True)
#trainGenerator = trainDatagen.flow(input_image, Y, batch_size=32, shuffle=True, seed=2,subset='training')
valDatagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.05,
        height_shift_range=0.05,
        #zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        #rescale = 1/255,
        #cval=0
        )
valGenerator = valDatagen.flow(X_val, Y_val, batch_size=32, shuffle=True)
#valGenerator = trainDatagen.flow(input_image, Y, batch_size=32, shuffle=True, seed=2,subset='validation')

size = 256
baseModel = keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(None, None, 3))
baseModel.trainable = False

class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'channels_last':
            self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)

        if self.dim_ordering == 'channels_first':
            num_rows = input_shape[2]
            num_cols = input_shape[3]
        elif self.dim_ordering == 'channels_last':
            num_rows = input_shape[1]
            num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        if self.dim_ordering == 'channels_first':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]
                        x_crop = x[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

        elif self.dim_ordering == 'channels_last':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix * col_length[pool_num]
                        x2 = ix * col_length[pool_num] + col_length[pool_num]
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]

                        x1 = K.cast(K.round(x1), 'int32')
                        x2 = K.cast(K.round(x2), 'int32')
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]

                        x_crop = x[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs


myModel = Sequential()
#myModel.add(GaussianNoise(stddev=stddev, input_shape=(160, 160, 3)))
myModel.add(baseModel)
#myModel.add(Conv2D(1024, 1))
#myModel.add(GlobalAveragePooling2D())
myModel.add(SpatialPyramidPooling([1, 2, 4]))
myModel.add(Dropout(0.2))
myModel.add(Dense(512, activation='relu'))
myModel.add(Dense(512, activation='relu'))
myModel.add(Dense(1, activation='sigmoid'))

myModel = multi_gpu_model(myModel,gpus=2) #使用几张显卡n等于几
#model_parallel.compile(...) #注意是model_parallel，不是model

myModel.summary()

#densenetmodel = load_model('F:/rice_heading/new_model/densnet121nobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})
vgg16model = load_model('F:/rice_heading/new_model/vgg16nobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})
resnetmodel = load_model('F:/rice_heading/new_model/resnet101nobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})
inceptionmodel = load_model('F:/rice_heading/new_model/inceptionV2nobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})
mobilemodel = load_model('F:/rice_heading/new_model/mobilenobx.h5',custom_objects={'SpatialPyramidPooling':SpatialPyramidPooling})

#densenetPred = densenetmodel.predict(input_image)
vgg16Pred = vgg16model.predict(input_image)
resnetPred = resnetmodel.predict(input_image)
inceptionPred = inceptionmodel.predict(input_image)
mobilePred = mobilemodel.predict(input_image)

#np.savetxt('F:/rice_heading/new_model/densenet.txt',densenetPred,fmt='%0.8f')
np.savetxt('F:/rice_heading/new_model/vgg16.txt',vgg16Pred,fmt='%0.8f')
np.savetxt('F:/rice_heading/new_model/resnet.txt',resnetPred,fmt='%0.8f')
np.savetxt('F:/rice_heading/new_model/inception.txt',inceptionPred,fmt='%0.8f')
np.savetxt('F:/rice_heading/new_model/mobile.txt',mobilePred,fmt='%0.8f')

tbCallBack = TensorBoard(
        log_dir = './log/1',
        write_graph = True, 
        write_images = True,
#        write_grads = True,
#        histogram_freq = 1
        )

model_checkpoint = ModelCheckpoint('densnet121.h5', monitor='val_loss', verbose=1, save_best_only=True)
csv_logger = CSVLogger('./log/densnet121.log')

myModel.compile(
        loss='mean_absolute_error',
        optimizer=keras.optimizers.adam(lr=0.00001),
        metrics=['mse'])

myModel.fit_generator(
        trainGenerator,
        #steps_per_epoch = len(trainGenerator),
        steps_per_epoch = 20,
        epochs=200,
        validation_data=valGenerator,
        #validation_steps = len(valGenerator),
        validation_steps = 20,
        shuffle = True,
        #validation_split=0.2,
        callbacks=[tbCallBack, model_checkpoint, csv_logger])
