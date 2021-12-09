# General
import random
import gym
from numpy.lib.polynomial import RankWarning
import pandas as pd
import numpy as np
# Neural network 
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import Adam
# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys
import signal
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

model = keras.models.load_model('../weight/all_contact_duration')
print(model.summary())
np. set_printoptions(threshold=np.inf)


def obtain_weights(model):
    weights = model.get_weights()

    print("dimension: ", weights[4].shape)

    # weights connecting input layer to hidden layer 1
    print("Input Layer -> Hidden Layer")
    print("dimension: ", weights[0].shape)
    print(weights[0])
    

    # bias of the hidden layer 1
    # print(weights[1])

    # weights connecting hidden layer 1 to the output layer
    print("Hidden Layer -> Output Layer")
    print("dimension: ", weights[2].shape)
    print(weights[2])
    
    # bias of the output layer
    # print(weights[3])

# obtain_weights(model)

weights = model.get_weights()
print(weights[0].shape)
print(weights[1].shape)
print(weights[2].shape)
print(weights[3].shape)
# print(weights[4].shape)
# print(weights[5].shape)
tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    layer_range=None
)

# fig, axs = plt.subplots(2, 2)

uniform_data1 = np.random.rand(10, 12)
uniform_data2 = np.random.rand(2, 12)
uniform_data3 = np.random.rand(10, 12)
uniform_data4 = np.random.rand(10, 12)

# fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[4,1,0.2]))

# print(weights[4])
plt.figure(1)
ax1 = sns.heatmap(weights[0])
plt.figure(2)
ax2 =sns.heatmap(weights[2])
# plt.figure(3)
# ax3 =sns.heatmap(weights[4])


plt.show()
