# coding: utf-8
# usage: python rtd_v5.py [TXT: annotation]
# txt == [path] [x] [y]
# description: In this version, I also considered sequence. Therefore, I implemented input layer for three input images

# measure time
import time
start = time.time()

# packages
import os
import sys
import re
import random

import os
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Keras outputs warnings using `print` to stderr so let's direct that to devnull temporarily
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
# we're done
sys.stderr = stderr
from keras.backend.tensorflow_backend import tf
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.setLevel(logging.FATAL)

####################
#     optional     #
####################
print_switcher = True
print_switcher_model = True
output_model = False

####################
#      STORAGE     #
####################
def data_reading(file, train_or_test, seq_len=3, batch_size=4, print_switcher=True):
    img_data_store = {}

    if print_switcher: print("################################")
    if train_or_test == "train":
        if print_switcher: print("#       TRAIN DATA READING     #")
    elif train_or_test == "test":
        if print_switcher: print("#        TEST DATA READING     #")
    else:
        if print_switcher: print("#       ????? DATA READING     #")
    if print_switcher: print("################################")
    
    with open(file, 'r') as f:
            dataline = f.readlines()
    for ann in dataline:
        if ' ' in ann:
            path, x, y = ann.split()
        else:
            path = ann.replace("\n", "")
        group_name = path.split("/")[0] + "/" + path.split("/")[1]
        if group_name in list(img_data_store.keys()):
            pass
        else:
            # name.add(path.split("/")[0])
            img_data_store[group_name] = []
        img_data_store[group_name].append(path.split("/")[2])
    
    available_num = 0
    
    for key in list(img_data_store.keys()):
        original_length = len(img_data_store[key])
        available_num += len(img_data_store[key]) - int((seq_len - 1))
        if print_switcher: print("parent_dir:\"" + key +"\",", "GROUP:", re.search(r"[^0-9]+", key).group(), ",", "NUM:", len(img_data_store[key]), "(available", str(len(img_data_store[key]) - int((seq_len - 1))) + ")")
        if len(img_data_store[key]) < seq_len:
            print("NOT ENOUGH IMG FILES IN THE DIRECTORY")
            exit()
    
    checker = img_data_store.copy()
    for key in checker.keys():
        checker[key] = checker[key][1:-1]
    # for key in list(checker.keys()):
    #     print(key)
    
    full_batch = []
    batch = []
    mini_batch = []
    flg = False
    
    while True:
        judger = 0
        for parent in list(checker.keys()):
            if len(checker[parent]) != 0:
                #print(checker[parent])
                judger += 1
        if judger == 0:
            break
        
        full_path_length = 0
        
        while len(batch) < batch_size:
            for key in list(checker.keys()):
                full_path_length += len(checker[key])
                if len(checker[key]) == 0:
                    checker.pop(key)
            
            parent, paths = random.choice(list(checker.items()))
            #print(len(paths))
            
            if full_path_length + len(mini_batch) < batch_size:
                flg = True
                break
    
            item_num = random.randrange(0, len(paths))
            item = checker[parent][item_num]
            if item in checker[parent]:
                checker[parent].remove(item)
                piv = int(img_data_store[parent].index(item))
                for i in range(-1, 2):
                    mini_batch.append(parent + "/" + img_data_store[parent][piv + i])
                # checker[parent].remove(item)
                # for i in range(-int(seq_len - 1), 1):
                #     mini_batch.append(parent + "/" + img_data_store[parent][item_num + i])
                
            batch.append(mini_batch)
            mini_batch = []
        
        if flg:
            break
        
        #print(batch)
        full_batch.append(batch)
        batch = []
    
    ann_data_store = {}
    with open(file, 'r') as f:
            dataline = f.readlines()
    for ann in dataline:
        if ' ' in ann:
            path, x, y = ann.split()
        else:
            path = ann.replace("\n", "")
            x = 641
            y = 361
        if path in list(ann_data_store.keys()):
            pass
        else:
            ann_data_store[path] = []
        ann_data_store[path].append(float(x))
        ann_data_store[path].append(float(y))
    #print(ann_data_store)
    
    return full_batch, ann_data_store, batch_size, available_num

####################
# CNN CONSTRUCTION #
####################

if print_switcher: print("################################")
if print_switcher: print("#        CNN CONSTRUCTION      #")
if print_switcher: print("################################")

# 学習準備
from keras.models import Model
from keras.layers import Input, Reshape, Permute, BatchNormalization, Activation

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model

# functional API model construction
inputs = Input(shape=(360, 640, 9))

# 1
x = ZeroPadding2D(1)(inputs)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 2
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 3
x = MaxPooling2D((2,2), strides=(2,2))(x)

# 4
x = ZeroPadding2D((1,1))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 5
x = ZeroPadding2D((1,1))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 6
x = MaxPooling2D((2,2), strides=(2,2))(x)

# 7
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 8
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 9
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 10
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 11
x = MaxPooling2D((2,2), strides=(2,2))(x)

# 12
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 13
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 14
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 15
x = ZeroPadding2D((1,1))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 16
x = UpSampling2D(size=(2, 2), data_format=None)(x)

# 17
x = ZeroPadding2D((1,1))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 18
x = ZeroPadding2D((1,1))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 19
x = UpSampling2D(size=(2, 2), data_format=None)(x)

# 20
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 21
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 22
x = UpSampling2D(size=(2, 2), data_format=None)(x)

# 23
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 24
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 25
x = ZeroPadding2D((1,1))(x)
x = Conv2D(1, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)

# 26
x = Reshape((230400, 1), input_shape=(360, 640, 1))(x)
x = Permute((2, 1))(x)
x = Activation("softmax")(x)
x = Permute((2, 1))(x)
x = Reshape((360, 640, 1))(x)

# o_shape = Model(inputs, x ).output_shape
# print(o_shape)

# define model
model_functional = Model(inputs=inputs, outputs=x, name="RTD_Net_v5")
# model_functional.outputWidth = OutputWidth
# model_functional.outputHeight = OutputHeight
if print_switcher_model: model_functional.summary()
if output_model: plot_model(model_functional, to_file="model.png", show_shapes=True)

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

#model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
# learning scheduler
# def step_decay(epoch):
#     x = 0.01
#     if epoch >= 40: x = 0.001
#     if epoch >= 80: x = 0.0001
#     return x
# lr_decay = LearningRateScheduler(step_decay)

model_functional.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

W_Check = os.path.isfile("weights.h5")
if W_Check:
    model_functional.load_weights(os.path.join("weights.h5"))
else:
    pass


####################
#     LEARNING     #
####################

if print_switcher: print("################################")
if print_switcher: print("#            LEARNING          #")
if print_switcher: print("################################")

t_available_num = 0
v_available_num = 0
t_full_batch, t_ann_data_store, batch_size, t_available_num = data_reading(sys.argv[1], train_or_test="train", print_switcher=True, seq_len=3)

# output all image num
print('alle:', t_available_num + v_available_num, 'train:', t_available_num, '+ valid:', v_available_num)

import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import bivariate_normal
import bivariate_uniform

# creating batch
def batch_iter(full_batch, ann_dict):
    while True:
        for batch in full_batch:
            seq = []
            ####################
            #      fix X       #
            ####################
            centre_path = []
            for mini_batch in batch:
                arrlist = []
                for num, img_path in enumerate(mini_batch):
                    if len(mini_batch) - int(len(mini_batch)/2) -1 == num:
                        centre_path.append(img_path)
                    img = load_img(img_path, target_size=(360, 640))
                    array = img_to_array(img) / 255
                    arrlist.append(array)
                arrlist = np.array(arrlist)
                arrlist = np.dstack(arrlist)
                seq.append(arrlist)
            X = np.array(seq)
            #print(X.shape)
            
            ####################
            #      fix Y       #
            ####################
            ann_list = []
            for path in centre_path:
                ann_list.append(ann_dict[path])
    
            y = []
            for num, i in enumerate(ann_list):
                Z = [[0] * 640 for _ in range(360)]
                # s, t = np.mgrid[0:135, 0:240]
                if int(i[0])==641 and int(i[1])==461:
                    #Z = bivariate_uniform.bivariate_uniform(360, 640)
                    Z = [[1/230400] * 640 for _ in range(360)]
                else:
                    for j in range(360):
                        for k in range(640):
                            Z[j][k] = bivariate_normal.bivariate_normal(k, j, sigmax=2.0, sigmay=2.0, mux=float(i[0]),
                                                                        muy=float(i[1]), sigmaxy=0.0)
                y.append(Z)
            Y = np.array(y)
            Y = Y.reshape(len(Y), 360, 640, 1)
            #print(Y.shape)
            tuple = (X, Y)
            #print(X.shape, y.shape)
            # time.sleep(5)
            yield tuple

from keras.callbacks import ModelCheckpoint
modelCheckpoint = ModelCheckpoint(filepath ='weights.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  period=1)

history = model_functional.fit_generator(generator=batch_iter(t_full_batch, t_ann_data_store),
                    steps_per_epoch=int(t_available_num/batch_size),
                    #validation_steps=int(v_available_num/batch_size),
                    epochs=100,
                    callbacks=[modelCheckpoint],
                    # validation_data=batch_iter(x_test, y_test, batch_size))
                    )
#model.save_weights('param.hdf5')

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# 学習結果を描写
import matplotlib.pyplot as plt

#loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# #loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()