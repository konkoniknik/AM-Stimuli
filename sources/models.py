import tensorflow as tf
import sys,os
import numpy as np
import time
import sources.help_functions as h
import argparse
import random
import time
import matplotlib.pyplot as plt

n_classes=10

def create_generator():
    model = tf.keras.Sequential()

    # creating Dense layer with units 7*7*256(batch_size) and input_shape of (100,)
    model.add(tf.keras.layers.Dense(512, use_bias=True, input_shape=(512,)))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(3920, use_bias=True ))#100
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.2))

    #Transpose convloutions
    model.add(tf.keras.layers.Reshape((7, 7, 80)))


    model.add(tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=True))
    model.add(tf.keras.layers.LeakyReLU())


    for _ in range(3):
        model.add(tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=(1, 1), padding='same', use_bias=True))
        model.add(tf.keras.layers.LeakyReLU())


    model.add(tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='same', use_bias=True))

    #model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x,0,1))) #Want to clip instead of sigmoid? (more unstable)
    model.add(tf.keras.layers.Activation('sigmoid'))



    return model


def create_Classifier(soft=0):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation=tf.nn.leaky_relu,padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation=tf.nn.leaky_relu,padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),


    tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), activation=tf.nn.leaky_relu,padding="same"),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), activation=tf.nn.leaky_relu,padding="same"),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(n_classes)])

  if(soft==1): model.add(tf.keras.layers.Softmax())
  return model

