import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras import Sequential, Model
from keras.activations import softmax
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPool2D


def orientation_model(n_classes, fc_layer_size=None):
    """Add last layer to the convnet

    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        new keras model with last layer
    """
    base_model = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    base_model_output = base_model.output
    # x = MaxPooling2D()(base_model_output)
    x = Dropout(0.4)(base_model_output)
    model_flat = Flatten()(base_model_output)
    predictions = Dense(n_classes, activation='softmax', name='predictions')(model_flat)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def damage_model(n_classes, fc_layer_size):
    """Add last layer to the convnet

    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        new keras model with last layer
    """
    base_model = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    base_model_output = base_model.output
    x = Dropout(0.4)(base_model_output)
    model_flat = Flatten()(base_model_output)
    predictions = Dense(n_classes, activation='sigmoid', name='predictions')(model_flat)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model 

