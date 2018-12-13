
# coding: utf-8

# In[5]:


import keras
from keras.constraints import max_norm

from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam
from keras.models import Sequential, Model 
from keras.layers import Reshape, BatchNormalization,Activation, Dropout, Dense, Conv2DTranspose, MaxPooling2D, MaxPooling1D, AveragePooling1D,Flatten, Dense, GlobalAveragePooling2D, Input, Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras import regularizers
import os 
import pandas as pd 
import numpy as np 
import shutil 


# # Start here: download the Imagenet style version of the skin cancer image dataset

# In[6]:


os.chdir("./Music")


# In[7]:


trainpath = "./cancerfolders/train"; valpath = "./cancerfolders/valid"
testpath = "./cancerfolders/testing"
import math 
batch_size = 32

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.05,
width_shift_range = 0.05,
height_shift_range=0.05)

val_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.05,
width_shift_range = 0.05,
height_shift_range=0.05)

train_generator = train_datagen.flow_from_directory(
    directory = trainpath,
    target_size=(224, 224),
    color_mode = "rgb",
    batch_size = 32,
    class_mode = "categorical",
    shuffle = True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    directory = valpath,
    target_size=(224, 224),
    color_mode = "rgb",
    batch_size = 32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

nb_train_samples = len(train_generator.filenames)  
num_classes = len(train_generator.class_indices)  
   
predict_size_train = int(math.ceil(nb_train_samples / batch_size))

nb_validation_samples = len(val_generator.filenames)  
   
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))  
   


# ## DenseNet Models 

# In[8]:


base_model = DenseNet169(include_top = False, weights = 'imagenet', input_tensor = None, input_shape=(224,224,3), pooling = None)


# In[9]:


for layer in base_model.layers[:]:
    layer.trainable = True
#print(len(base_model.layers))


# In[17]:


#Adding custom Layers total layers = 431
from keras.regularizers import l2 
x = base_model.output
x = Dropout(0.2)(x)
#x = Dense(512, activation = "relu", 
      #   kernel_initializer = "he_normal")(x)
#x = Dropout(0.1)(x)
#x = Dense(64, activation = "relu", kernel_initializer = "he_normal")(x)
x = Flatten()(x)
#x = Dropout(0.1)(x)
x = Dense(7, activation = "softmax", 
          kernel_initializer = "he_normal")(x)



# creating the final model 
model = Model(inputs = base_model.input, outputs = x)
model.summary()


# In[ ]:


model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=1e-3, clipnorm = 0.05), metrics=["accuracy"])


checkpoint = ModelCheckpoint("densenet169.h5", monitor='val_acc', verbose=1, 
                             save_best_only = True, 
                             save_weights_only = False, 
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', 
                      min_delta = 1e-4, 
                      patience = 10, 
                      verbose=1, mode='auto') 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.9, 
                              patience = 3, min_lr = 1e-5, verbose = 1)



history = model.fit_generator(train_generator, epochs = 100, validation_data = val_generator, 
                              verbose = 1, steps_per_epoch = nb_train_samples // batch_size, 
                              validation_steps = nb_validation_samples // batch_size, 
                              callbacks = [reduce_lr])


# # Extras and Utilities

# In[8]:


import tensorflow as tf


def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl


# In[4]:


import tensorflow as tf 
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))


# ## Resnet152 Model 

# In[12]:


# -*- coding: utf-8 -*-
"""ResNet152 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adaptation of code from flyyufelix, mvoelk, BigMoyan, fchollet

"""

import numpy as np
import warnings

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import add
from keras.models import Model
import keras.backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import obtain_input_shape

import sys
sys.setrecursionlimit(3000)

WEIGHTS_PATH = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/adamcasson/resnet152/releases/download/v0.1/resnet152_weights_tf_notop.h5'

class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    eps = 1.1e-5
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def ResNet152(include_top=True, weights=None,
              input_tensor=None, input_shape=None,
              large_input=False, pooling=None,
              classes=1000):
  
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    
    eps = 1.1e-5
    
    if large_input:
        img_size = 448
    else:
        img_size = 224
    
    # Determine proper input shape
    input_shape = (224,224,3)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # handle dimension ordering for different backends
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if large_input:
        x = AveragePooling2D((14, 14), name='avg_pool')(x)
    else:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
    
    # include classification layer by default, not included for feature extraction 
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet152')
    
    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet152_weights_tf.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='cdb18a2158b88e392c0905d47dcef965')
        else:
            weights_path = get_file('resnet152_weights_tf_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='4a90dcdafacbd17d772af1fb44fc2660')
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
                
        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    return model


# In[20]:


rn152 = ResNet152(include_top = False, input_shape = (224,224,3), weights = "imagenet")
y = rn152.output
y = Dropout(0.8)(y)
y = Dense(512, activation = "relu", kernel_constraint=max_norm(2))(y)
y = Flatten()(y)
y = Dropout(0.8)(y)
y = Dense(7, activation = "softmax", kernel_constraint=max_norm(2))(y)


# In[28]:


rn152model = Model(inputs = rn152.input, outputs = y)
rn152model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=1e-2, clipnorm = 1.), metrics=["accuracy"])


checkpoint = ModelCheckpoint("rn152cancer_model.h5", monitor='val_acc', verbose=1, 
                             save_best_only = True, 
                             save_weights_only = False, 
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', 
                      min_delta = 1e-4, 
                      patience = 10, 
                      verbose=1, mode='auto') 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.9, 
                              patience = 3, min_lr = 1e-5, verbose = 1)



rn152model_history = rn50model.fit_generator(train_generator, epochs = 100, validation_data = val_generator, 
                              verbose = 1, steps_per_epoch = nb_train_samples // batch_size, 
                              validation_steps = nb_validation_samples // batch_size, 
                              callbacks = None)


# ## ResNet50 model 

# In[13]:


from keras.applications.resnet50 import ResNet50
resmodel = ResNet50(include_top = False, weights = "imagenet", input_shape = (224,224,3))


# In[11]:



#Adding custom Layers total layers = 431
x = resmodel.output
x = Dropout(0.5)(x)
x = Dense(512, activation = "relu", kernel_initializer = "he_normal")(x)
x = Dropout(0.5)(x)
x = Dense(32, activation = "relu", kernel_initializer = "he_normal")(x)
x = Flatten()(x)
x = Dense(7, activation = "softmax", kernel_initializer = "he_normal")(x)



# creating the final model 
rn50model = Model(inputs = resmodel.input, outputs = x)
rn50model.summary()

rn50model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr=1e-3, clipnorm = 1.), metrics=["accuracy"])


checkpoint = ModelCheckpoint("rn50cancer_model.h5", monitor='val_acc', verbose=1, 
                             save_best_only = True, 
                             save_weights_only = False, 
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', 
                      min_delta = 1e-4, 
                      patience = 10, 
                      verbose=1, mode='auto') 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.9, 
                              patience = 3, min_lr = 1e-5, verbose = 1)



rn50model_history = rn50model.fit_generator(train_generator, epochs = 100, validation_data = val_generator, 
                              verbose = 1, steps_per_epoch = nb_train_samples // batch_size, 
                              validation_steps = nb_validation_samples // batch_size, 
                              callbacks = None)

