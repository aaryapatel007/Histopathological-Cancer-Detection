import os
import numpy as np
import cv2
import glob
import gc
import itertools
import matplotlib.pyplot as plt
from albumentations import (Resize,Compose, RandomRotate90, Transpose, Flip, OneOf, CLAHE, IAASharpen, IAAEmboss,
RandomBrightnessContrast, JpegCompression, Blur, GaussNoise, HueSaturationValue, ShiftScaleRotate, Normalize)
from random import shuffle

def get_id_from_file_path(file_path):
    return file_path.split(os.path.sep)[-1].replace('.tif', '')


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def do_train_augmentations():
    return Compose([
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        Flip(p=0.5),
        OneOf([CLAHE(clip_limit=2),
              IAASharpen(),
              IAAEmboss(),
              RandomBrightnessContrast(),
              JpegCompression(),
              Blur(),
              GaussNoise()],
              p=0.5),
        HueSaturationValue(p=0.5),
        ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
        Normalize(p=1)])


def do_inference_aug():
    return Compose([Normalize(p=1)], p=1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('octresnet_cm.png')
  
def data_gen(list_files, id_label_map_in, batch_size_in, img_size_in, aug_funtion):
  
    aug = aug_funtion()
    
    while True:
      
        shuffle(list_files)
        for block in chunker(list_files, batch_size_in):

            X = [cv2.resize(cv2.imread(x), (img_size_in, img_size_in)) for x in block]
            X = [aug(image=x)['image'] for x in X]

            Y = [id_label_map_in[get_id_from_file_path(x)] for x in block]

            yield np.array(X), np.array(Y)

