import cv2
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model,load_model
from keras.applications.densenet import DenseNet201, preprocess_input,decode_predictions
# pip install keras-vis
from vis.utils import utils
from vis.visualization import visualize_cam

#plots gradCAM visualization
def plot_map(img, grads, class_index, y_pred):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(img)
    axes[1].imshow(img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    plt.suptitle("Pr(class={}) = {:5.6f}".format(
                      class_label[class_index],
                      y_pred[0,0]))
    plt.show()
    plt.savefig(class_label[class_index] + '.png')

if __name__ == "__main__":
  #read the train data CSV file
  df = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

  #making independent lists for tumor and non-tumor tissues
  list_no_tumor = df.loc[df['label'] == 0]['id'].tolist()
  list_tumor = df.loc[df['label'] == 1]['id'].tolist()

  #load keras model
  init_model = load_model('../input/densenet-8020/densenet169_one_cycle_model.h5')

  class_label = ['no_tumor','tumor']

  random_int = np.random.choice(len(list_tumor))
  img_no_tumor = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + list_no_tumor[random_int] + '.tif')
  img_tumor = load_img('/kaggle/input/histopathologic-cancer-detection/train/' + list_tumor[random_int] + '.tif')

  img_no_tumor = img_to_array(img_no_tumor)
  img_no_tumor = preprocess_input(img_no_tumor)
  y_pred_no_tumor = init_model.predict(img_no_tumor[np.newaxis,...])

  img_tumor = img_to_array(img_tumor)
  img_tumor = preprocess_input(img_tumor)
  y_pred_tumor = init_model.predict(img_tumor[np.newaxis,...])

  layer_idx = utils.find_layer_idx(init_model, 'dense_3')
  # Swap softmax with linear
  init_model.layers[layer_idx].activation = keras.activations.linear
  model = utils.apply_modifications(init_model)

  penultimate_layer_idx = utils.find_layer_idx(model, "relu") 

  seed_input = img_no_tumor
  grad_top1_no_tumor  = visualize_cam(model, layer_idx, 0, seed_input, 
                             penultimate_layer_idx = penultimate_layer_idx,#None,
                             backprop_modifier = None,
                             grad_modifier = None)


  seed_input = img_tumor
  grad_top1_tumor  = visualize_cam(model, layer_idx, 0, seed_input, 
                             penultimate_layer_idx = penultimate_layer_idx,#None,
                             backprop_modifier = None,
                             grad_modifier = None)

  plot_map(img_no_tumor, grad_top1_no_tumor, 0, y_pred_no_tumor)
  plot_map(img_tumor, grad_top1_tumor, 1, y_pred_tumor)
