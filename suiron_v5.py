# coding: utf-8
# usage: python suiron_v2.py [img]
# description v1: used sequential
# description v2: used Functional API model and implemented skip connection

import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import cv2

from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPooling2D, add, UpSampling2D, Dense, Reshape, BatchNormalization, Activation, Permute

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
outputs = Reshape((360, 640, 1))(x)

model_functional = Model(inputs=inputs, outputs=outputs, name="VGG_like_model")

arrlist = []
for num, path in enumerate(sys.argv):
    if num != 0:
        img = load_img(path)
        array = img_to_array(img) / 255
        arrlist.append(array)
arrlist = np.array(arrlist)
arrlist = np.dstack(arrlist)
array = arrlist.reshape(1, 360, 640, 9)

model_functional.load_weights('weights_20210330.h5')
out = model_functional.predict(array)

sum = 0
highest_coord_x = 0
highest_coord_y = 0
highest_p = 0
# ALL
for i in out:
    # Y
    for c_Y, j in enumerate(i):
        # X
        for c_X, k in enumerate(j):
            for l in k:
                sum += l
                # sum_coord_y += c_Y * l
                # sum_coord_x += c_X * l
                if max(l, highest_p)==l:
                    highest_p=l
                    highest_coord_x = c_X
                    highest_coord_y = c_Y
print(highest_coord_x, highest_coord_y, highest_p)
print(sum)
# print([sum_coord_x, sum_coord_y])

out = out.reshape(360, 640)
#sns.heatmap(out,cmap='Blues',annot=True,square = True,fmt="1.5f",vmax=1.0,cbar = True)
#sns.heatmap(out,cmap='Blues',square = True,fmt="1.5f",vmax=1.0)
grid_kws = {"height_ratios": (.9, .1), "hspace": .5}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

idx = np.unravel_index(np.argmax(out), out.shape)

max_peak = np.max(out)
size = out.shape

img = np.full((360, 640, 3), 128, dtype=np.uint8)
for i in range(size[0]):
    for j in range(size[1]):
        rate = out[i][j]/max_peak
        cv2.rectangle(img, (j, i), ((j+1), (i+1)), (int(rate*255), int(rate*255), 20), thickness=-1)
        # cv2.rectangle(img, 左上座標, 右下座標, BGR, thickness=-1で内部塗りつぶし)
cv2.imwrite('tmp.png', img)

sns.heatmap(out,cmap='Blues',square = True, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
plt.savefig('seaborn_heatmap_list.png')
#plt.close('all')
