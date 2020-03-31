# Training code for simple hold-out train-val split
import pandas as pd
from random import shuffle
import numpy as np
import cv2
import glob
import gc
import os
import tensorflow as tf

from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Add, GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,concatenate
from keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN, Callback
from keras.optimizers import Adam,RMSprop,SGD
from keras.models import Model,load_model
from keras.applications import NASNetMobile,MobileNetV2,densenet,resnet50,xception
from keras_applications.resnext import ResNeXt50

from albumentations import Resize,Compose, RandomRotate90, Transpose, Flip, OneOf, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, JpegCompression, Blur, GaussNoise, HueSaturationValue, ShiftScaleRotate, Normalize

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from skimage import data, exposure
import itertools
import shutil
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from one_cycle_policy_lr import *
from data_generator import *
from Model.se import *
from Model.OctaveResNet import *
from Model.ResNeXt import *
from Model.ResNeXt_CBAM import *
from Model.se_resnext import *
from Model.baseline_model import *

# Training Hyperparameters and other variables
img_size = (96,96,3)
batch_size = 192
epochs = 38

# reading the training data CSV file
df = pd.read_csv("/CSV/train_labels.csv")
df_train, df_val = train_test_split(df, test_size=0.1, stratify= df['label'])

print("Train data: " + str(len(df_train[df_train["label"] == 1]) + len(df_train[df_train["label"] == 0])))
print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))
print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))
print("Valid data: " + str(len(df_val[df_val["label"] == 1]) + len(df_val[df_val["label"] == 0])))
print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))
print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))

# Train List
train_list = df_train['id'].tolist()
train_list = ['/path/to/dir/'+ name + ".tif" for name in train_list]

# Validation List
val_list = df_val['id'].tolist()
val_list = ['/path/to/dir'+ name + ".tif" for name in val_list]

# Dictionary mapping Image IDs to corresponding labels....used in data_generator.py
id_label_map = {k:v for k,v in zip(df.id.values, df.label.values)}

# Using octaveresnet50 for training.
#We can use different models such as ResNeXt50, Seresnet50 by replacing them.

#base_model = ResNextImageNet(include_top=False, weights=None,  input_shape=img_size)
base_model = OctaveResNet50(input_shape=img_size, include_top=False,
                           alpha=0.5, expansion=4,
                           initial_filters=64,
                           initial_strides=False)
x = base_model.output

out1 = GlobalMaxPooling2D()(x)
out2 = GlobalAveragePooling2D()(x)
#out3 = Flatten()(x)
out = concatenate([out1,out2])
out = BatchNormalization(epsilon = 1e-5)(out)
out = Dropout(0.4)(out)
fc = Dense(512,activation = 'relu')(out)
fc = BatchNormalization(epsilon = 1e-5)(fc)
fc = Dropout(0.3)(fc)
fc = Dense(256,activation = 'relu')(fc)
fc = BatchNormalization(epsilon = 1e-5)(fc)
fc = Dropout(0.3)(fc)
X = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')(fc)
model =  Model(inputs=base_model.input, outputs=X)

# lr_callback = LRFinder(len(train_list), batch_size,
#                        1e-5, 1.,
#                        # validation_data=(X_val, Y_val),
#                        lr_scale='exp', save_dir='weights/')

# The best values of max_lr are found by running one epoch of LRFinder on the whole data.
lr_manager = OneCycleLR(max_lr=0.02, end_percentage=0.1, scale_percentage=None,
                        maximum_momentum=0.9,minimum_momentum=0.8)    
callbacks = [lr_manager,
           ModelCheckpoint(filepath='octresnet_one_cycle_model.h5', monitor='val_loss',mode='min',verbose=1,save_best_only=True)]

model.compile(loss='binary_crossentropy', optimizer=SGD(0.002, momentum=0.9, nesterov=True), metrics=['accuracy'])

history = model.fit_generator(data_gen(train_list, id_label_map, batch_size,do_train_augmentations),
                              validation_data=data_gen(val_list, id_label_map, batch_size,do_inference_aug),
                              epochs = epochs,
                              steps_per_epoch = (len(train_list) // batch_size) + 1,
                              validation_steps = (len(val_list) // batch_size) + 1,
                              callbacks=callbacks,
                              verbose = 1)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "valid"], loc="upper left")
plt.savefig('loss_performance.png')
plt.clf()
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.title("model acc")
plt.ylabel("acc")
plt.xlabel("epoch")
plt.legend(["train", "valid"], loc="upper left")
plt.savefig('acc_performance.png')
