import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Input, Concatenate
import time

from tensorflow.keras.models import Model

class ppo_network():
  def __init__(self, input_size1, input_size2, input_size3, action_size, hidden_sizes1, \
                      hidden_sizes2, alpha=2e-4, beta=4e-4):
    self.action_size = action_size
    self.latter_struct = hidden_sizes2
    self.former_struct = hidden_sizes1
    self.input_size1 = input_size1
    self.input_size2 = input_size2
    self.input_size3 = input_size3
    self.act_type = 'relu'
    self.alpha=alpha
    self.policy_opt = keras.optimizers.Adam(learning_rate=alpha)
    self.value_opt = keras.optimizers.Adam(learning_rate=beta)
    self.policy, self.critic = self.build_net()
  
  def build_net(self):
    # Initialize the policy models    
    policy = self.policy_network(input1=self.input_size1, input2=self.input_size2, input3=self.input_size3, 
                          hid_layers_1=self.former_struct, hid_layers_2=self.latter_struct)
    # Initialize the critic model
    critic = self.critic_network(input1=self.input_size1, input2=self.input_size2, input3=self.input_size3,
                          hid_layers_1=self.former_struct, hid_layers_2=self.latter_struct, activationType='tanh')
    return policy, critic

  def policy_network(self, input1, input2, input3, hid_layers_1, hid_layers_2, activationType='tanh'):
    
    x_1 = Input(shape=(input1,))
    x_2 = Input(shape=(input2,))
    x_3 = Input(shape=(input3,)) 


    for index in range(len(hid_layers_1)):
      if index == 0:
        layer_i = Dense(hid_layers_1[index], activation='relu')(x_1)
      else:
        layer_i = Dense(hid_layers_1[index], activation='relu')(layer_i)
    
    y = Model(inputs=x_1, outputs=layer_i)
    
    combined = Concatenate()([y.output, x_2, x_3])

    for ii in range(len(hid_layers_2)):
      if ii == 0:
        z = Dense(hid_layers_2[ii],activation='relu')(combined)
      else:
        z = Dense(hid_layers_2[ii],activation='relu')(z)

    mean_linear = Dense(1, activation='sigmoid')(z)
    mean_rot = Dense(1, activation='tanh')(z)
    # std_linear = Dense(1, activation='sigmoid')(z)
    # std_rot = Dense(1, activation='sigmoid')(z)
    std_linear = Dense(1, activation='linear')(z)
    std_rot = Dense(1, activation='linear')(z)
    policy = Model(inputs=[x_1, x_2, x_3], outputs=[mean_linear, mean_rot, std_linear, std_rot])
    policy.summary()
    return policy


  def critic_network(self, input1, input2, input3, hid_layers_1, hid_layers_2, activationType='tanh'):
    x_1 = Input(shape=(input1,))
    x_2 = Input(shape=(input2,))
    x_3 = Input(shape=(input3,)) 

    for index in range(len(hid_layers_1)):
      if index == 0:
        layer_i = Dense(hid_layers_1[index], activation='relu')(x_1)
      else:
        layer_i = Dense(hid_layers_1[index], activation='relu')(layer_i)
    
    y = Model(inputs=x_1, outputs=layer_i)
    
    combined = Concatenate()([y.output, x_2, x_3])

    for ii in range(len(hid_layers_2)):
      if ii == 0:
        z = Dense(hid_layers_2[ii],activation='relu')(combined)
      else:
        z = Dense(hid_layers_2[ii],activation='relu')(z)

    state_value = Dense(1, activation='linear')(z)
    critic = Model(inputs=[x_1, x_2, x_3], outputs=state_value)
    # policy.summary()

    critic.summary()
    return critic

class ppo_agent():
  def __init__(self, input1, input2, input3, action_space, hidden_sizes1, hidden_sizes2,\
                    clip_ratio=0.15, path='/content/', model_name='ppo'):
    # self.action_space = action_space
    self.n_actions = action_space
    self.path = path
    self.model_name=model_name
    self.clip_ratio = clip_ratio
    self.ppo_net = ppo_network(input_size1=input1, input_size2=input2, input_size3=input3, 
                action_size=self.n_actions, hidden_sizes1=hidden_sizes1, hidden_sizes2=hidden_sizes2)

  def log_probs(self, logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    log_probs_all = tf.nn.log_softmax(logits)
    log_prob = tf.reduce_sum(tf.one_hot(a, self.n_actions) * log_probs_all, axis=1)
    return log_prob
  
  # Sample action from actor
  def select_action(self, observation):
    mean_linear, mean_rot, std_linear, std_rot = self.ppo_net.policy(observation)

    # be sure the stds are positive
    sigma_linear = tf.exp(std_linear)
    sigma_rot = tf.exp(std_rot)
    # set up the distribution functions
    action_probs_linear = tf.distributions.Normal(mean_linear, sigma_linear)
    action_probs_rot = tf.distributions.Normal(mean_rot, sigma_rot)
    
    # get probabilities from the distributions
    prob_1 = action_probs_linear.sample(1)
    prob_2 = action_probs_rot.sample(1)
    
    # get log probabilities 



    # action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

 
  # Train the policy by maxizing the PPO-Clip objective
  # def train(self, observations, actions, log_probs, advantages, returns):
  #   # for policy
  #   with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
  #       ratio = tf.exp(
  #           self.log_probs(self.ppo_net.actor(observations), actions) - log_probs)
  #       min_advantage = tf.where(
  #           advantages > 0,
  #           (1 + self.clip_ratio) * advantages,
  #           (1 - self.clip_ratio) * advantages, )
  #       policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))

  #   policy_grads = tape.gradient(policy_loss, self.ppo_net.actor.trainable_variables)
  #   self.ppo_net.policy_opt.apply_gradients(zip(policy_grads, self.ppo_net.actor.trainable_variables))

  #   # Train the value function by regression on mean-squared error
  #   with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
  #       value_loss = tf.reduce_mean((returns - self.ppo_net.critic(observations)) ** 2)
  #   value_grads = tape.gradient(value_loss, self.ppo_net.critic.trainable_variables)
  #   self.ppo_net.value_opt.apply_gradients(zip(value_grads, self.ppo_net.critic.trainable_variables))