# Training code for stratified K-Fold train-val split
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
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report, roc_curve, auc
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
train_batch_size = 224
val_batch_size = 224
n_folds = 5
epochs = 38

# reading the training data CSV file
df_train = pd.read_csv("/CSV/train_labels.csv")

# Dictionary mapping Image IDs to corresponding labels....used in data_generator.py
id_label_map = {i:j for i,j in zip(df_train.id.values,df_train.label.values)}

train_files = glob.glob('../path/to/dir/*.tif')
test_files = glob.glob('../path/to/dir/*.tif')

print("train_files size :", len(train_files))
print("test_files size :", len(test_files))

df_dataset = pd.DataFrame()
df_dataset['id'] = '../path/to/dir/' + df_train['id'] + '.tif'
df_dataset['label'] = df_train['label']

# remove corrupted images
df_dataset = df_dataset[df_dataset['id'] != '../path/to/dir/dd6dfed324f9fcb6f93f46f32fc800f2ec196be2.tif']
df_dataset = df_dataset[df_dataset['id'] != '../path/to/dir/9369c7278ec8bcc6c880d99194de09fc2bd4efbe.tif']


ensemble_preds = np.zeros(len(test_files), dtype=np.float)
skf = StratifiedKFold(n_splits=n_folds)

# Start K-Fold training
for fold in range(n_folds):

    filepath = 'resnext_cbam_model_' + str(fold) + '.h5'

    print("\nFOLD: {}".format(fold))

    result = next(skf.split(df_dataset,df_dataset['label']))

    train = df_dataset.iloc[result[0]]['id'].values.tolist()
    val = df_dataset.iloc[result[1]]['id'].values.tolist()

    train_steps = len(train) // train_batch_size
    val_steps = len(val) // val_batch_size

    print("Train: ")
    values,count = np.unique(df_dataset.iloc[result[0]]['label'].values,return_counts = True)
    print(dict(zip(values,count)))

    print("Val: ")
    values,count = np.unique(df_dataset.iloc[result[1]]['label'].values,return_counts = True)
    print(dict(zip(values,count)))
    
    # Defining ResNeXt_CBAM model
    base_model = ResNextImageNet(include_top=False, weights=None,  input_shape=img_size)
    x = base_model.output

    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    #out3 = Flatten()(x)
    out = concatenate([out1,out2])
    out = BatchNormalization(epsilon = 1e-5)(out)
    fc = Dropout(0.4)(out)
    fc = Dense(256,activation = 'relu')(fc)
    fc = BatchNormalization(epsilon = 1e-5)(fc)
    fc = Dropout(0.3)(fc)
    X = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros')(fc)
    model =  Model(inputs=base_model.input, outputs=X)
    
    # Compiling the Keras Model
    model.compile(loss='binary_crossentropy', optimizer=SGD(0.002, momentum=0.99, nesterov=True), metrics=['accuracy'])
    
    lr_manager = OneCycleLR(max_lr=0.02, end_percentage=0.1, scale_percentage=None,
                            maximum_momentum=0.99,minimum_momentum=0.89)

    callbacks = [lr_manager,
               ModelCheckpoint(filepath=filepath, monitor='val_loss',mode='min',verbose=1,save_best_only=True)]

    # Training Begins
    history = model.fit_generator(data_gen(train,id_label_map,train_batch_size,do_train_augmentations),steps_per_epoch=train_steps,epochs = 9,
                                   validation_data = data_gen(val,id_label_map,val_batch_size,do_inference_aug),validation_steps = val_steps,callbacks = callbacks)



    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('loss_performance'+'_'+str(fold)+'.png')
    plt.clf()
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='valid')
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('acc_performance'+'_'+str(fold)+'.png')
    del history
    del model
    gc.collect()
