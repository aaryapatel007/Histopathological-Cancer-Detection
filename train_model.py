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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN
from keras.optimizers import Adam,RMSprop,SGD
from keras.models import Model,load_model
from keras.applications import NASNetMobile,MobileNetV2,densenet,resnet50,xception

from keras_applications.resnext import ResNeXt50
from albumentations import Resize,Compose, RandomRotate90, Transpose, Flip, OneOf, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, JpegCompression, Blur, GaussNoise, HueSaturationValue, ShiftScaleRotate, Normalize


from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold
from skimage import data, exposure
import itertools
import shutil
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


df_train, df_val = train_test_split(df, test_size=0.1, stratify= df['label'])

print("Train data: " + str(len(df_train[df_train["label"] == 1]) + len(df_train[df_train["label"] == 0])))
print("True positive in train data: " +  str(len(df_train[df_train["label"] == 1])))
print("True negative in train data: " +  str(len(df_train[df_train["label"] == 0])))
print("Valid data: " + str(len(df_val[df_val["label"] == 1]) + len(df_val[df_val["label"] == 0])))
print("True positive in validation data: " +  str(len(df_val[df_val["label"] == 1])))
print("True negative in validation data: " +  str(len(df_val[df_val["label"] == 0])))
