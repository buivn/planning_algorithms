#!/usr/bin/env python

# author: Hoang-Dung Bui
# email: hbui20@gmu.edu
# Description: In this program, there is 1 dynamic obstacle in the environment, and actor network provide two outputs:
# the linear velocity from 0.0- 1.0 (10 ranges) and rotational velocity (-pi - pi: 21 ranges)

import gym
from gym import wrappers
import gym_gazebo
import liveplot
import json
import os


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from tensorflow.keras.models import load_model
from ppo_v1 import ppo_agent
# from per_v1 import Buffer_v0
from per_v1 import Buffer_nonPer

if __name__ == '__main__':
  tf.enable_eager_execution()
  # --------------------------INITIALIZE ALL VALUES OF:...............................#
  agent_number = 1        # define the number of agent
  outdir = '/tmp/gazebo_gym_experiments/'
  env = gym.make('Dynamic_Obstacles-v0')
  plotter = liveplot.LivePlot(outdir)
  continue_execution = False
  # Hyperparameters of the PPO algorithm
  if not continue_execution:
    #Each time we take a sample and update our weights it is called a mini-batch.
    #Each time we run through the entire dataset, it's called an epoch.
    #PARAMETER LIST
    epochs = 300
    max_steps = 300
    # updateTargetNetwork = 10000
    explorationRate = 1
    batch_size = 64
    learnStart = 64
    learningRate = 0.0002
    discountFactor = 0.99
    action_dims = 2
    mem_size = 150000
    network_inputs = 100
    network_outputs1 = 10
    network_outputs2 = 21
    network_structure = [512, 512, 512]
    current_epoch = 0
    train_iterations = 3 

  # Initialize the buffer
  buffer2 = Buffer_nonPer(network_inputs, action_dims, mem_size)
  # buffer2 = Buffer_v0(network_inputs, action_dims, mem_size)
  # input_size, action_space1, action_space2, hidden_sizes
  agent = ppo_agent(network_inputs, network_outputs1, network_outputs2, network_structure)
  # agent.ppo_net.actor = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_actor_dyn_v1_512x3.h5')
  # agent.ppo_net.critic = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_critic_dyn_v1_512x3.h5')
  # print(agent.ppo_net.actor.summary())
  eps_min = 0.01
  eps_dec = 5e-6
  epsilon = 1.0 
  
  env._max_episode_steps = max_steps # env returns done after _max_episode_steps
  env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)
  start_time = time.time()

  episode_return = 0
  episode_length = 0
  observation = env.reset()
  last100Scores = [0] * 100
  last100ScoresIndex = 0
  last100Filled = False

  demo = False

  # Iterate over the number of epochs
  for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    episode_return = 0.0
    episode_length = 0
    done = False
    # Iterate until it reached target or a max step
    while not done:
      # Get the logits, action, and take one step in the environment
      observation = observation.reshape(1, -1)        # adding 1 more dimension
      # logits_linear, logits_rot, action_linear, action_rot = agent.select_action(observation)
      logits_linear, logits_rot, action_linear, action_rot = agent.select_action_phil(observation)
      act1 = action_linear.numpy()

      act2 = action_rot.numpy()
      # combine two action into a vector
      act = np.concatenate([act1, act2])

      # print(act)
      observation_new, reward, done, _ = env.step(act)
      episode_return += reward
      episode_length += 1

      # Get the value and log-probability of the action
      value_t = agent.ppo_net.critic(observation).numpy()[0]
      # log_prob_t = agent.log_probs(logits_linear, logits_rot, act1, act2)
      log_prob_t = [logits_linear, logits_rot]
      # Store obs, act, rew, v_t, logp_pi_t
      buffer2.store_experience(observation, act, reward, value_t, log_prob_t)
      # buffer2.store_experience(observation, act, reward, value_t, [logits_linear, logits_rot])
      
      # Update the observation
      observation = observation_new.copy()

      # Finish trajectory if reached to a terminal state
      if done:
        last_value = 0 if done else agent.ppo_net.critic(observation.reshape(1, -1))
        buffer2.finish_trajectory(last_value)
        observation = env.reset()
        break
    
    if not demo:
      for _ in range(15):
        if buffer2.pointer > batch_size:
          # Get values from the buffer
          # tree_idx, obsers, actions, advantages, returns, log_probs_b = buffer2.get_sample(batch_size)
          obsers, actions, advantages, returns, log_probs_b = buffer2.get_sample(batch_size)
          # Update the policy and implement early stopping using KL divergence      
          for _ in range(train_iterations):
            abs_err = agent.train(obsers, actions, advantages, returns, log_probs_b)
            # buffer2.batch_update(tree_idx, abs_err)

    last100Scores[last100ScoresIndex] = episode_length
    last100ScoresIndex += 1
    if last100ScoresIndex >= 100:
        last100Filled = True
        last100ScoresIndex = 0

    if not last100Filled:
        print ("EP " + str(epoch) + " - " + format(episode_length + 1) + "/" + str(max_steps) + " Episode steps " )
    else :    
      m, s = divmod(int(time.time() - start_time), 60)
      h, m = divmod(m, 60)
      print ("EP " + str(epoch) + " - " + format(episode_length + 1) + "/" + str(max_steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(episode_return) + "     Time: %d:%02d:%02d" % (h, m, s))
      # if (epoch)%100==0:
      #     env._flush()
    


    if (epoch+1) %100 == 0:
      plotter.plot(env)
      if not demo:
        agent.ppo_net.actor.save('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_actor_dyn_v1_512x3_256.h5')
        agent.ppo_net.critic.save('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_critic_dyn_v1_512x3_256.h5')
        print("Save the model successfully")

  env.close()

