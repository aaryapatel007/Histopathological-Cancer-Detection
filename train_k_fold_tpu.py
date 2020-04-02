import numpy as np
import pandas as pd

import gc
import os
from glob import glob
from random import shuffle
import cv2
import datetime
import matplotlib.pyplot as plt

from albumentations import (Compose, RandomRotate90, 
Transpose, Flip, OneOf, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, JpegCompression, Blur, GaussNoise,
 HueSaturationValue, ShiftScaleRotate, Normalize)

from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Add, GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,concatenate
from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras import Input
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications import NASNetMobile,MobileNetV2,densenet,resnet50,xception

from one_cycle_policy_lr import *
from data_generator import *
from Model.se import *
from Model.OctaveResNet import *
from Model.ResNeXt import *
from Model.ResNeXt_CBAM import *
from Model.se_resnext import *
from Model.baseline_model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_metric(history_in, metric_name, results_dir):
    """
    Plot a metric of model's history.
    """

    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history_in.history[metric_name])
    plt.plot(history_in.history['val_' + metric_name])

    plt.title('model ' + metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_acc.savefig(results_dir+"/model_" + metric_name + ".png")

    plt.cla()
    plt.close()


def mkdir_if_not_exist(directory):

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))



train_limit = 220025 
test_limit = 57458 
epochs = 10 
n_fold = 2 
number_of_tpu_core = 8  
batch_size = 64

if __name__ == '__main__':
  
    ###############
    # Configuration
    ###############
    img_size = 96
    training_batch_size = batch_size * number_of_tpu_core
    
    
    ##################
    # Data Preparation
    ##################
    
    df_train = pd.read_csv("/content/train_labels.csv")
    id_label_map = {k: v for k, v in zip(df_train.id.values, df_train.label.values)}

    labeled_files =glob. glob('/content/train/*.tif')
    test_files = glob.glob('/content/test/*.tif')

    print("labeled_files size :", len(labeled_files))
    print("test_files size :", len(test_files))

    df_dataset = pd.DataFrame()
    df_dataset['id'] = labeled_files[0:train_limit]
    df_dataset['label'] = df_train['label'].iloc[0:train_limit]

    ensemble_preds = np.zeros(len(test_files[0:test_limit]), dtype=np.float)

    skf = StratifiedKFold(n_splits=n_fold)

    for fold in range(0, n_fold):

        print("\nFOLD: {}".format(fold))

        mkdir_if_not_exist(os.path.join(output_path, str(fold)))

        h5_path = output_path + "/model_"+str(fold)+".h5"

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                verbose=1,
                restore_best_weights=True),
            ModelCheckpoint(
                h5_path,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='min')]

        result = next(skf.split(X=df_dataset, y=df_dataset['label']), None)

        train = df_dataset.iloc[result[0]]['id'].tolist()
        val = df_dataset.iloc[result[1]]['id'].tolist()

        print("train:")
        unique, counts = np.unique(df_dataset.iloc[result[0]]['label'].values, return_counts=True)
        print(dict(zip(unique, counts)))

        print("val:")
        unique, counts = np.unique(df_dataset.iloc[result[1]]['label'].values, return_counts=True)
        print(dict(zip(unique, counts)))
        
        ######################
        # Put the model on TPU
        ######################
        
        tf.keras.backend.clear_session()
                   
        # This address identifies the TPU we'll use when configuring TensorFlow.
        TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        tf.logging.set_verbosity(tf.logging.INFO)
        
        #
        res_model = get_model_resnet50(img_size,batch_size)
        
        # Converting the Keras model to TPU model using tf.contrib.tpu.keras_to_tpu_model
        model = tf.contrib.tpu.keras_to_tpu_model(
            res_model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))

        model.summary()

        ###########
        # Training
        ###########

        history = model.fit_generator(
            data_gen(train, id_label_map, training_batch_size, img_size, do_train_augmentations),
            validation_data=data_gen(val, id_label_map, training_batch_size, img_size, do_inference_aug),
            epochs=epochs, verbose=1,
            callbacks=callbacks,
            steps_per_epoch=len(train) // training_batch_size,
            validation_steps=len(val) // training_batch_size)

        # summarize history
        plot_metric(history, 'loss', str(fold))
        plot_metric(history, 'acc', str(fold))

        
