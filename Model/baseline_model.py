from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input, BatchNormalization, Add, GlobalAveragePooling2D,AveragePooling2D,GlobalMaxPooling2D,concatenate
from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.models import Model,load_model

def simple_custom_model(img_size, batch_size):
  
    weight_decay = 1e-4

    visible = Input(shape=(img_size, img_size, 3), batch_size=batch_size, dtype=tf.float32)

    conv1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(visible)
    conv1_act = Activation('elu')(conv1)
    conv1_act_batch = BatchNormalization()(conv1_act)

    conv2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(conv1_act_batch)
    conv2_act = Activation('elu')(conv2)
    conv2_act_batch = BatchNormalization()(conv2_act)
    conv2_act_batch_max = MaxPooling2D(pool_size=(2, 2))(conv2_act_batch)
    conv2_act_batch_max_drop = Dropout(0.2)(conv2_act_batch_max)

    conv3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(
        conv2_act_batch_max_drop)
    conv3_act = Activation('elu')(conv3)
    conv3_act_batch = BatchNormalization()(conv3_act)

    conv4 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(conv3_act_batch)
    conv4_act = Activation('elu')(conv4)
    conv4_act_batch = BatchNormalization()(conv4_act)
    conv4_act_batch_max = MaxPooling2D(pool_size=(2, 2))(conv4_act_batch)
    conv4_act_batch_max_drop = Dropout(0.3)(conv4_act_batch_max)

    conv5 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(
        conv4_act_batch_max_drop)
    conv5_act = Activation('elu')(conv5)
    conv5_act_batch = BatchNormalization()(conv5_act)

    conv6 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(conv5_act_batch)
    conv6_act = Activation('elu')(conv6)
    conv6_act_batch = BatchNormalization()(conv6_act)
    conv6_act_batch_max = MaxPooling2D(pool_size=(2, 2))(conv6_act_batch)
    conv6_act_batch_max_drop = Dropout(0.4)(conv6_act_batch_max)
    
    flat = Flatten()(conv6_act_batch_max_drop)

    # and a logistic layer
    predictions = Dense(1, activation='sigmoid')(flat)
    
    # Create model.
    model = tf.keras.Model(visible, predictions, name='baseline')

    model.compile(optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy', metrics=['acc'])

    return model

if __name__ == "__main__":
  inputs = Input(shape=(96, 96, 3), batch_size=512)
  base_model = get_model(96,512)
  print(base_model.summary())
