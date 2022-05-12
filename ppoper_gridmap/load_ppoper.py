import sys, os
import random
from tkinter import *
import time
import math, queue
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import Dense, Input, Flatten
import tensorflow.keras.backend as K
import pyglet  # this package control the display window


class Agent():
  def __init__(self, path):
    self.actor_path = path

  def load(self):
    self.actor = load_model(self.actor_path, compile=False)
