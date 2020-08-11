#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:03:31 2020

@author: adonay
"""
import numpy as np
import math
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Conv1D,
                                     MaxPooling1D, Input, concatenate)
import tensorflow as tf
from tensorflow.keras import Input, layers


def model_1D_Seq(X, y, ker_sz1=10, ker_sz2=10, n_ker1=40, n_ker2=40, n_flat=100):
    n_timesteps, n_features, n_outputs = (X.shape[1], X.shape[2],
                                          y.shape[1])
    model = Sequential()
    model.add(Conv1D(filters=n_ker1, kernel_size=ker_sz1, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=n_ker2, kernel_size=ker_sz2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(n_flat, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    return model


def model_1d_low_high(X, y):
    i_shape = X.shape[1:3]
    drop_out_rate = 0.5
    input_shape = i_shape
    
    input_tensor = Input(shape=(input_shape))

    x = layers.Conv1D(8, 11, padding='valid', activation='relu',
                      strides=1)(input_tensor)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(drop_out_rate)(x)
    output_tensor = layers.Dense(y.shape[1], activation='softmax')(x)

    model = tf.keras.Model(input_tensor, output_tensor)
    return model

def layer_viz(model):
    # summarize filter shapes
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        
        # get filter weights
        filters, biases = layer.get_weights()
        n_filt = filters.shape[-2:]
        print(layer.name, filters.shape)

        fig, axs = plt.subplots(n_filt[0], n_filt[1], sharex='col',
                                sharey='row', gridspec_kw={'hspace': 0})

        fil_flat = np.reshape(filters, [-1, np.prod(n_filt) ])
        for ax, fil in zip(axs.flat, fil_flat.T):
            ax.plot(fil.T)
        fig.suptitle(f"Name: {layer.name}, shape: {filters.shape}")

