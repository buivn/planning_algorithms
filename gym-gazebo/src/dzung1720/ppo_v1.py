# author: Hoang-Dung Bui
# email: hbui20@gmu.edu/bui.hoangdungtn@gmail.com
# Description: In this rl networks, the actor network outputs two items: rotational and linear velocities (discrete)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input, Concatenate
import time

import tensorflow_probability as tfp
from tensorflow.keras.models import Model

class ppo_network():
  def __init__(self, input_size, action_size1, action_size2, hidden_sizes, \
                      alpha=3e-4, beta=4e-4):
    self.action_size1 = action_size1
    self.action_size2 = action_size2
    self.hidden_sizes = hidden_sizes
    self.input_size = input_size
    self.act_type = 'relu'
    self.alpha=alpha
    self.policy_opt = keras.optimizers.Adam(learning_rate=alpha)
    self.value_opt = keras.optimizers.Adam(learning_rate=beta)
    self.actor, self.critic = self.build_net()
  
  def build_net(self):
    # Initialize the actor models
    actor = self.policy_network(input1=self.input_size, hid_layers=self.hidden_sizes, 
              outputs1=self.action_size1, outputs2=self.action_size2)

    # Initialize the critic model
    critic = self.critic_network(input1=self.input_size, hid_layers=self.hidden_sizes)
    return actor, critic

  def policy_network(self, input1, hid_layers, outputs1, outputs2):    
    x_1 = Input(shape=(input1,))
    for index2 in range(len(hid_layers)):
      if index2 == 0:
        linear_dense = Dense(hid_layers[index2], activation='relu')(x_1)
        rot_dense = Dense(hid_layers[index2], activation='relu')(x_1)
      else:
        linear_dense = Dense(hid_layers[index2], activation='relu')(linear_dense)
        rot_dense = Dense(hid_layers[index2], activation='relu')(rot_dense)

      
    linear = Dense(outputs1, activation='softmax')(linear_dense)  
    # rot_dense = Dense(hid_layers[-1], activation='relu')(rot_dense1)
    rot = Dense(outputs2, activation='softmax')(rot_dense)

    policy = Model(inputs=x_1, outputs=[linear, rot])
    # policy.summary()
    return policy

  def critic_network(self, input1, hid_layers):    
    x_1 = Input(shape=(input1,))
    for index in range(len(hid_layers)):
      if index == 0:
        layer_i = Dense(hid_layers[index], activation='relu')(x_1)
      else:
        layer_i = Dense(hid_layers[index], activation='relu')(layer_i)
    
    # value_dense = Dense(156, activation='relu')(layer_i)
    value = Dense(1, activation=None)(layer_i)

    critic = Model(inputs=x_1, outputs=value)
    # critic.summary()
    return critic


class ppo_agent():
  def __init__(self, input_size, action_space1, action_space2, hidden_sizes, \
                    clip_ratio=0.2, path='/content/', model_name='ppo'):

    self.n_actions1 = action_space1
    self.n_actions2 = action_space2
    self.path = path
    self.model_name=model_name
    self.clip_ratio = clip_ratio
    self.ppo_net = ppo_network(input_size=input_size, action_size1=self.n_actions1, \
                                    action_size2=self.n_actions2, hidden_sizes=hidden_sizes)

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
    # log_prob = tf.add(log_prob2,log_prob1)

    return [log_prob1, log_prob2] #log_prob
  
  # Sample action from actor
  def select_action(self, observation):
    (logits_linear, logits_rot) = self.ppo_net.actor(observation)
    linear = tf.squeeze(tf.random.categorical(logits_linear, 1), axis=1)
    rot = tf.squeeze(tf.random.categorical(logits_rot, 1), axis=1)
    return logits_linear, logits_rot, linear, rot


  # Sample action from actor
  def select_action_phil(self, observation):
    (logits_linear, logits_rot) = self.ppo_net.actor(observation)
    dist_linear = tfp.distributions.Categorical(logits_linear)
    dist_rot = tfp.distributions.Categorical(logits_rot)
    
    linear = dist_linear.sample()
    log_prob_linear = dist_linear.log_prob(linear)

    rot = dist_rot.sample()
    log_prob_rot = dist_rot.log_prob(rot)
    
    # linear = linear.numpy()[0]
    log_prob_linear = log_prob_linear.numpy()[0]

    # rot = rot.numpy()[0]
    log_prob_rot = log_prob_rot.numpy()[0]
    
    # return logits_linear, logits_rot, linear, rot
    return log_prob_linear, log_prob_rot, linear, rot


  # Train the policy by maxizing the PPO-Clip objective
  def train(self, obsers, actions, advantages, returns, log_probs_b):
    with tf.GradientTape(persistent=True) as tape:  # Record operations for automatic differentiation.
      log_linear, log_rot = self.ppo_net.actor(obsers)
      
      dist_linear = tfp.distributions.Categorical(log_linear)
      dist_rot = tfp.distributions.Categorical(log_rot)
      # linear = dist_linear.sample()
      new_log_prob_linear = dist_linear.log_prob(actions[:,0])

      # rot = dist_rot.sample()
      new_log_prob_rot = dist_rot.log_prob(actions[:,1])
      # diff1, diff2 = self.log_probs(log_linear, log_rot, actions[:,0], actions[:,1])
      
      critic_values = self.ppo_net.critic(obsers)
      critic_values = tf.squeeze(critic_values,1)
      diff1 = new_log_prob_linear-log_probs_b[:,0]
      diff2 = new_log_prob_rot-log_probs_b[:,1]
      prob_ratio_linear = tf.math.exp(diff1)
      prob_ratio_rot = tf.math.exp(diff2)


      clipped_probs_1 = tf.clip_by_value(
                                        prob_ratio_linear,
                                        (1 + self.clip_ratio),
                                        (1 - self.clip_ratio))

      clipped_probs_2 = tf.clip_by_value(
                                        prob_ratio_rot,
                                        (1 + self.clip_ratio),
                                        (1 - self.clip_ratio))
      
      weighted_probs_1 = prob_ratio_linear * advantages
      weighted_probs_2 = prob_ratio_rot * advantages

      # get the average
      policy_loss1 = -tf.math.reduce_mean(tf.math.minimum(clipped_probs_1 * advantages, weighted_probs_1))
      policy_loss2 = -tf.math.reduce_mean(tf.math.minimum(clipped_probs_2 * advantages, weighted_probs_2))
    
      value_loss = tf.reduce_mean((returns - critic_values) ** 2)
      

    # update the priorities of the data
    diff_sum = tf.add(diff1, diff2)
    abs_err = np.abs(diff_sum.numpy().squeeze())

    # train the actor
    policy_grads = tape.gradient([policy_loss1, policy_loss2], self.ppo_net.actor.trainable_variables)
    self.ppo_net.policy_opt.apply_gradients(zip(policy_grads, self.ppo_net.actor.trainable_variables))


    # with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
    #     value_loss = tf.reduce_mean((returns - self.ppo_net.critic(obsers)) ** 2)
    
    # Train the value function by regression on mean-squared error
    value_grads = tape.gradient(value_loss, self.ppo_net.critic.trainable_variables)
    self.ppo_net.value_opt.apply_gradients(zip(value_grads, self.ppo_net.critic.trainable_variables))
    return abs_err