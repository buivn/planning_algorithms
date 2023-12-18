import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras import Sequential, optimizers, initializers
from tensorflow.keras.layers import Dense, Activation, Input
import time

# import tensorflow_probability as tfp
from tensorflow.keras.models import Model

class ppo_network():
  def __init__(self, input_size, action_size, hidden_sizes, \
                      alpha=2e-4, beta=4e-4):
    self.action_size = action_size
    self.hidden_sizes = hidden_sizes
    self.input_size = input_size
    self.act_type = 'relu'
    self.alpha=alpha
    self.policy_opt = keras.optimizers.Adam(learning_rate=alpha)
    self.value_opt = keras.optimizers.Adam(learning_rate=beta)
    self.actor, self.critic = self.build_net()
  
  def build_net(self):
    # Initialize the actor models    
    actor = self.createModel(inputs=self.input_size, outputs=self.action_size,
                          hiddenLayers=self.hidden_sizes, activationType='tanh', learningRate=self.alpha)
    # Initialize the critic model
    critic = self.createModel(inputs=self.input_size, outputs=1,
                          hiddenLayers=self.hidden_sizes, activationType='tanh', learningRate=self.alpha)
    return actor, critic

  def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
    model = Sequential()
    if len(hiddenLayers) == 0:
        model.add(Dense(self.output_size, input_shape=(self.input_size,)))
        model.add(Activation("linear"))
    else :
        # model.add(Dense(input_shape=(self.input_size,)))
        model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,)))
        # model.add(Dense(inputs, kernel_initializer='lecun_uniform'))
        if (activationType == "LeakyReLU") :
            model.add(LeakyReLU(alpha=0.01))
        else :
            model.add(Activation(activationType))

        for index in range(1, len(hiddenLayers)):
            # print("adding layer "+str(index))
            layerSize = hiddenLayers[index]
            model.add(Dense(layerSize, kernel_initializer='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
        model.add(Dense(outputs, kernel_initializer='lecun_uniform'))
        # if outputs == 1:
        model.add(Activation("linear"))
        # else:
        #   model.add(Activation("softmax"))
    # optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
    # optimizer = optimizers.Adam(lr=learningRate)
    # model.compile(loss="mse", optimizer=optimizer)
    model.summary()
    return model


class ppo_agent():
  def __init__(self, input_size, action_space, hidden_sizes, \
                    clip_ratio=0.1, path='/content/', model_name='ppo'):
    # self.action_space = action_space
    self.n_actions = action_space
    self.path = path
    self.model_name=model_name
    self.clip_ratio = clip_ratio
    self.ppo_net = ppo_network(input_size=input_size, action_size=self.n_actions, hidden_sizes=hidden_sizes)

  def log_probs(self, logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    # print("Start the data -------------------------------------------")
    # print(logits)
    log_probs_all = tf.nn.log_softmax(logits)
    # print(log_probs_all)
    # tf.one_hot create a one hot tensor
    # print(tf.one_hot(a, self.n_actions))
    log_prob = tf.reduce_sum(tf.one_hot(a, self.n_actions) * log_probs_all, axis=1)
    # print(log_prob)
    return log_prob
  
  # Sample action from actor
  def select_action(self, observation):
    logits = self.ppo_net.actor(observation)
    # print(logits)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action



class ppo_network1():
  def __init__(self, input_size, action_size, hidden_sizes, \
                      alpha=3e-4, beta=3e-4):
    self.action_size = action_size
    self.hidden_sizes = hidden_sizes
    self.input_size = input_size
    # self.act_type = 'relu'
    self.alpha=alpha
    self.policy_opt = keras.optimizers.Adam(learning_rate=alpha)
    self.value_opt = keras.optimizers.Adam(learning_rate=beta)
    self.linear =self.createModel(inputs=self.input_size, outputs=5,
                          hiddenLayers=self.hidden_sizes, act='linear', learningRate=self.alpha)
    self.rot =self.createModel(inputs=self.input_size, outputs=self.action_size,
                          hiddenLayers=self.hidden_sizes, act='linear', learningRate=self.alpha)

    self.critic = self.critic_network(inputs=self.input_size, hiddenLayers=self.hidden_sizes)
  
  def critic_network(self, inputs, hiddenLayers):    
    x_1 = Input(shape=(inputs,))
    # initializer = tf.initializers.HeNormal()
    for index in range(len(hiddenLayers)):
      if index == 0:
        layer_i = Dense(hiddenLayers[index], kernel_initializer='lecun_uniform', activation='tanh')(x_1)
      else:
        layer_i = Dense(hiddenLayers[index], kernel_initializer='lecun_uniform', activation='tanh')(layer_i)
    
    # value_dense = Dense(156, activation='relu')(layer_i)
    value = Dense(1, activation='linear')(layer_i)

    return Model(inputs=x_1, outputs=value)


  def createModel(self, inputs, outputs, hiddenLayers, act, learningRate):
    model = Sequential()
    if len(hiddenLayers) == 0:
        model.add(Dense(self.output_size, input_shape=(self.input_size,)))
        model.add(Activation("linear"))
    else :
        # model.add(Dense(input_shape=(self.input_size,)))
        model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,)))
        model.add(Activation('tanh'))

        for index in range(1, len(hiddenLayers)):
            layerSize = hiddenLayers[index]
            model.add(Dense(layerSize, kernel_initializer='lecun_uniform'))
            model.add(Activation('tanh'))
        model.add(Dense(outputs, kernel_initializer='lecun_uniform'))
        # if outputs == 1:
        model.add(Activation(act))
        # model.add(Activation(""))
    # model.summary()
    return model



class ppo_agent1(): # This class provide 3 NNs: linear, rotational velocities, and critic.
  def __init__(self, input_size, action_space, hidden_sizes, \
                    clip_ratio=0.12, path='/content/', model_name='ppo'):
    # self.action_space = action_space
    self.n_actions = action_space
    self.path = path
    self.model_name=model_name
    self.clip_ratio = clip_ratio
    self.ppo_net = ppo_network1(input_size=input_size, action_size=self.n_actions, hidden_sizes=hidden_sizes)

  def log_probs(self, logits_linear, logits_rot, action_linear, action_rot):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    log_probs_linear = tf.nn.log_softmax(logits_linear)
    log_probs_rot = tf.nn.log_softmax(logits_rot)
    # tf.one_hot create a one hot tensor
    log_prob_linear = tf.reduce_sum(tf.one_hot(action_linear, 5) * log_probs_linear, axis=1)
    log_prob_rot = tf.reduce_sum(tf.one_hot(action_linear, self.n_actions) * log_probs_rot, axis=1)
    
    return [log_prob_linear, log_prob_rot]
  
  # Sample action from actor
  def select_action(self, observation):
    logits_linear = self.ppo_net.linear(observation)
    logits_rot = self.ppo_net.rot(observation)
    # print(logits)
    action_linear = tf.squeeze(tf.random.categorical(logits_linear, 1), axis=1)
    action_rot = tf.squeeze(tf.random.categorical(logits_rot, 1), axis=1)
    return logits_linear, logits_rot, action_linear, action_rot
