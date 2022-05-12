import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

import tensorflow_probability as tfp
from tensorflow.keras.models import Model

class ppo_network():
  def __init__(self, input_size, action_size, hidden_sizes, \
                      alpha=3e-4, beta=1e-3):
    self.action_size = action_size
    self.hidden_sizes = hidden_sizes
    self.input_size = input_size
    # Initialize the policy and the value function optimizers
    self.policy_opt = keras.optimizers.Adam(learning_rate=alpha)
    self.value_opt = keras.optimizers.Adam(learning_rate=beta)
    self.actor, self.critic = self.build_net()
  
  def build_net(self):
    # define the input tensor
    input_size_t = keras.Input(shape=(self.input_size,), dtype=tf.float32)
    # Initialize the actor models    
    logits = self.mlp(input_size_t, list(self.hidden_sizes) + [self.action_size], 'relu', None)
    actor = keras.Model(inputs=input_size_t, outputs=logits)
    # Initialize the critic model
    value = tf.squeeze(
        self.mlp(input_size_t, list(self.hidden_sizes) + [1], 'relu', None), axis=1)
    critic = keras.Model(inputs=input_size_t, outputs=value)
    return actor, critic

  def mlp(self, x, sizes, activation='tanh', output_activation=None):
    # Build a feedforward layer
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


class ppo_agent():
  def __init__(self, input_size, action_space, hidden_sizes, \
                    clip_ratio=0.2, path='/conten/', model_name='ppo'):
    self.action_space = action_space
    self.n_actions = len(self.action_space)
    self.path = path
    self.model_name=model_name
    self.clip_ratio = clip_ratio
    self.ppo_net = ppo_network(input_size=input_size, action_size=self.n_actions, hidden_sizes=hidden_sizes)

  def log_probs(self, logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    log_probs_all = tf.nn.log_softmax(logits)
    log_prob = tf.reduce_sum(tf.one_hot(a, self.n_actions) * log_probs_all, axis=1)
    return log_prob
  
  # Sample action from actor
  def select_action(self, observation):
    logits = self.ppo_net.actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    # print("what's wrong here")
    return logits, action

  def select_action_2(self, observation):
    logits = self.ppo_net.actor(observation)
    if np.random.random() > epsilon:
      action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    else:
      logit_1 = tf.math.log([[1., 1., 1., 1.]])
      num_samples = 1
      action = tf.squeeze(tf.random.categorical(logit_1, num_samples), axis=1)
    return logits, action

  # Train the policy by maxizing the PPO-Clip objective
  def train(self, observations, actions, log_probs, advantages, returns):
    # for policy
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            self.log_probs(self.ppo_net.actor(observations), actions) - log_probs)
        min_advantage = tf.where(
            advantages > 0,
            (1 + self.clip_ratio) * advantages,
            (1 - self.clip_ratio) * advantages, )
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))

    policy_grads = tape.gradient(policy_loss, self.ppo_net.actor.trainable_variables)
    self.ppo_net.policy_opt.apply_gradients(zip(policy_grads, self.ppo_net.actor.trainable_variables))

    # Train the value function by regression on mean-squared error
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((returns - self.ppo_net.critic(observations)) ** 2)
    value_grads = tape.gradient(value_loss, self.ppo_net.critic.trainable_variables)
    self.ppo_net.value_opt.apply_gradients(zip(value_grads, self.ppo_net.critic.trainable_variables))