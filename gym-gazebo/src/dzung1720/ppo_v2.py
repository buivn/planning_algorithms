# author: Hoang-Dung Bui
# email: hbui20@gmu.edu/bui.hoangdungtn@gmail.com
# Description: In this rl networks, the actor network outputs two items: rotational and linear velocities (discrete).
# The inputs of the networks are the states of system, which consists: laser data, distance to goal, and agent's velocities.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input, Concatenate
import time

# import tensorflow_probability as tfp
from tensorflow.keras.models import Model

class ppo_network():
  def __init__(self, input_size1, input_size2, input_size3, action_size1, \
              action_size2, hidden_sizes1, hidden_sizes2, alpha=2e-4, beta=4e-4):
    self.action_size1 = action_size1
    self.action_size2 = action_size2
    self.hidden_sizes1 = hidden_sizes1
    self.hidden_sizes2 = hidden_sizes2
    self.input_size1 = input_size1
    self.input_size2 = input_size2
    self.input_size3 = input_size3
    self.act_type = 'relu'
    self.alpha=alpha
    self.policy_opt = keras.optimizers.Adam(learning_rate=alpha)
    self.value_opt = keras.optimizers.Adam(learning_rate=beta)
    self.actor, self.critic = self.build_net()
  
  def build_net(self):
    # Initialize the actor models
    actor = self.policy_network(input1=self.input_size1,input2=self.input_size1, input3=self.input_size1, \
         hid_layers1=self.hidden_sizes1, hid_layers2=self.hidden_sizes2, outputs1=self.action_size1, outputs2=self.action_size2)

    # Initialize the critic model
    critic = self.critic_network(input1=self.input_size1,input2=self.input_size1, input3=self.input_size1, \
                                    hid_layers1=self.hidden_sizes1, hid_layers2=self.hidden_sizes2)
    return actor, critic

  def policy_network(self, input1, input2, input3, hid_layers1, hid_layers2, \
                              outputs1, outputs2, activationType='tanh'):    
    x_1 = Input(shape=(input1,))
    x_2 = Input(shape=(input2,))
    x_3 = Input(shape=(input3,))

    for index in range(len(hid_layers1)):
      if index == 0:
        layer_i = Dense(hid_layers1[index], activation='relu')(x_1)
      else:
        layer_i = Dense(hid_layers1[index], activation='relu')(layer_i)

    y = Model(inputs=x_1, outputs=layer_i)
    
    combined = Concatenate()([y.output, x_2, x_3])

    for ii in range(len(hid_layers2)):
      if ii == 0:
        z = Dense(hid_layers2[ii],activation='relu')(combined)
      else:
        z = Dense(hid_layers2[ii],activation='relu')(z)

    linear = Dense(outputs1, activation='linear')(z)
    rot = Dense(outputs2, activation='linear')(z)

    policy = Model(inputs=[x_1,x_2,x_3], outputs=[linear, rot])
    policy.summary()
    return policy

  def critic_network(self, input1, input2, input3, hid_layers1, hid_layers2):    
    x_1 = Input(shape=(input1,))
    x_2 = Input(shape=(input2,))
    x_3 = Input(shape=(input3,))

    for index in range(len(hid_layers1)):
      if index == 0:
        layer_i = Dense(hid_layers1[index], activation='relu')(x_1)
      else:
        layer_i = Dense(hid_layers1[index], activation='relu')(layer_i)

    y = Model(inputs=x_1, outputs=layer_i)
    
    combined = Concatenate()([y.output, x_2, x_3])

    for ii in range(len(hid_layers2)):
      if ii == 0:
        z = Dense(hid_layers2[ii],activation='relu')(combined)
      else:
        z = Dense(hid_layers2[ii],activation='relu')(z)


    value = Dense(1, activation='linear')(z)

    critic = Model(inputs=[x_1,x_2,x_3], outputs=value)
    critic.summary()
    return critic


class ppo_agent():
  def __init__(self, input_size1, input_size2, input_size3, action_space1, action_space2, \
            hidden_sizes1, hidden_sizes2, clip_ratio=0.15, path='/content/', model_name='ppo'):

    self.n_actions1 = action_space1
    self.n_actions2 = action_space2
    self.path = path
    self.model_name=model_name
    self.clip_ratio = clip_ratio
    self.ppo_net = ppo_network(input_size1=input_size1, input_size2=input_size2, input_size3=input_size3,\
          action_size1=self.n_actions1, action_size2=self.n_actions2, hidden_sizes1=hidden_sizes1,hidden_sizes2=hidden_sizes2)

  def log_probs(self, logits1, logits2, action1, action2):
    # Compute the log-probabilities of taking actions a by using the logits
    # The log-softmax penalty has a exponential nature compared to the linear 
    # penalisation of softmax. i.e More heavy peanlty for being more wrong
    log_probs_1 = tf.nn.log_softmax(logits1)
    log_probs_2 = tf.nn.log_softmax(logits2)
    # tf.one_hot create a one hot tensor 
    one_hot1 =  tf.one_hot(action1, self.n_actions1)
    one_hot2 =  tf.one_hot(action2, self.n_actions2)
    # sum all elements of the tensor along axis
    log_prob1 = tf.reduce_sum(one_hot1 * log_probs_1, axis=1)
    log_prob2 = tf.reduce_sum(one_hot2*log_probs_2, axis=1)
    log_prob = tf.add(log_prob2,log_prob1)
    return log_prob
  
  # Sample action from actor
  def select_action(self, laser, distances, vel):
    (logits_linear, logits_rot) = self.ppo_net.actor(laser, distances, vel)
    linear = tf.squeeze(tf.random.categorical(logits_linear, 1), axis=1)
    rot = tf.squeeze(tf.random.categorical(logits_rot, 1), axis=1)
    return logits_linear, logits_rot, linear, rot


  # Train the policy by maxizing the PPO-Clip objective
  def train(self, laser, distances, vel, actions, advantages, returns, log_probs_b):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
      log_linear, log_rot = self.ppo_net.actor(laser, distances, vel)
      diff = self.log_probs(log_linear, log_rot, actions[:,0], actions[:,1]) - log_probs_b
      ratio = tf.exp(diff)
      min_advantage = tf.where(
          advantages > 0,
          (1 + self.clip_ratio) * advantages,
          (1 - self.clip_ratio) * advantages, )
      # get the average
      policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))
    # update the priorities of the data
    abs_err = np.abs(diff.numpy().squeeze())

    # train the actor
    policy_grads = tape.gradient(policy_loss, self.ppo_net.actor.trainable_variables)
    self.ppo_net.policy_opt.apply_gradients(zip(policy_grads, self.ppo_net.actor.trainable_variables))

    # Train the value function by regression on mean-squared error
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((returns - self.ppo_net.critic(laser, distances, vel)) ** 2)
    value_grads = tape.gradient(value_loss, self.ppo_net.critic.trainable_variables)
    self.ppo_net.value_opt.apply_gradients(zip(value_grads, self.ppo_net.critic.trainable_variables))
    return abs_err