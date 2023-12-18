#!/usr/bin/env python

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
# from env_bin import create_environment_binary
from ppo import ppo_agent
from per import Buffer_3ver

if __name__ == '__main__':
  # tf.compat.v1.disable_eager_execution()
  tf.enable_eager_execution()
  # --------------------------INITIALIZE ALL VALUES OF:...............................#
  agent_number = 1        # define the number of agent
  outdir = '/tmp/gazebo_gym_experiments/'
  env = gym.make('Static_Obstacles-v0')
  # env = gym.make('Dynamic_Obstacles-v0')
  plotter = liveplot.LivePlot(outdir)
  continue_execution = False
  # Hyperparameters of the PPO algorithm
  if not continue_execution:
    #Each time we take a sample and update our weights it is called a mini-batch.
    #Each time we run through the entire dataset, it's called an epoch.
    #PARAMETER LIST
    epochs = 2
    max_steps = 300
    # updateTargetNetwork = 10000
    # explorationRate = 1
    batch_size = 64
    learnStart = 64
    learningRate = 0.0002
    discountFactor = 0.99
    mem_size = 150000
    network_inputs = 100
    network_outputs = 21
    network_structure = [512, 512, 512]
    current_epoch = 0
    # clip_ratio = 0.2
    # policy_lr = 3e-4
    # value_function_lr = 1e-3
    train_iterations = 3
    lam = 0.97
    target_kl = 0.01
    agent_number = 1        # define the number of agent 

  # Initialize the buffer
  buffer2 = Buffer_3ver(network_inputs, mem_size)

  agent = ppo_agent(network_inputs, network_outputs, network_structure)
  agent.ppo_net.actor = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/1_output/ppoper_actor_512512512_1out_2dyn_ran.h5')
  agent.ppo_net.critic = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/1_output/ppoper_critic_512512512_1out_2dyn_ran.h5')
  # print(agent.ppo_net.actor.summary())
  # eps_min = 0.01
  # eps_dec = 5e-6
  # epsilon = 1.0 
  
  env._max_episode_steps = max_steps # env returns done after _max_episode_steps
  env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)
  start_time = time.time()

  # for pair in range(n_pairs_layer1):
    # buffer2.buffer_reset()
  
  # epsilon = 1.0
  episode_return = 0
  episode_length = 0
  observation = env.reset()
  last100Scores = [0] * 50
  last100ScoresIndex = 0
  last100Filled = False

  learning = False
  number_success = 0.0

  # Iterate over the number of epochs
  for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    episode_return = 0.0
    episode_length = 0
    done = False
    # Iterate until it reached target or a max step
    while not done:
      # Get the logits, action, and take one step in the environment
      # adding 1 more dimension
      observation = observation.reshape(1, -1)
      logits, action = agent.select_action(observation)
      act = action[0].numpy()
      # print(act)
      observation_new, reward, done, _ = env.step(act)
      episode_return += reward
      episode_length += 1

      # Get the value and log-probability of the action
      value_t = agent.ppo_net.critic(observation)
      log_prob_t = agent.log_probs(logits, action)

      # Store obs, act, rew, v_t, logp_pi_t
      buffer2.store_experience(observation, act, reward, value_t, log_prob_t)
      # epsilon = epsilon - eps_dec \
      #               if epsilon > eps_min else eps_min
      # Update the observation
      observation = observation_new.copy()

      # Finish trajectory if reached to a terminal state
      if done:
        last_value = 0 if done else agent.ppo_net.critic(observation.reshape(1, -1))
        buffer2.finish_trajectory(last_value)
        observation = env.reset()
        if episode_return > 10.0:
          number_success += 1.0
        break
    
    if learning:
      if 0.6 > float(number_success)/float(epoch+1) > 0.3 and epoch > 40:
        num_batch = 10
      elif float(number_success)/float(epoch+1) >= 0.6 and epoch > 40:
        num_batch = 20
      else:
        num_batch = 5
      for _ in range(num_batch):
        if buffer2.pointer > batch_size:
          # Get values from the buffer
          tree_idx, obsers, actions, advantages, returns, log_probs_b = buffer2.get_sample(batch_size)
          # Update the policy and implement early stopping using KL divergence      
          for _ in range(train_iterations):
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
              diff = agent.log_probs(agent.ppo_net.actor(obsers), actions) - log_probs_b
              ratio = tf.exp(diff)
              min_advantage = tf.where(
                  advantages > 0,
                  (1 + agent.clip_ratio) * advantages,
                  (1 - agent.clip_ratio) * advantages, )
              policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, min_advantage))
            # update the priorities of the data
            abs_err = np.abs(diff.numpy().squeeze())

            buffer2.batch_update(tree_idx, abs_err)
            # train the actor
            policy_grads = tape.gradient(policy_loss, agent.ppo_net.actor.trainable_variables)
            agent.ppo_net.policy_opt.apply_gradients(zip(policy_grads, agent.ppo_net.actor.trainable_variables))

            # Train the value function by regression on mean-squared error
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                value_loss = tf.reduce_mean((returns - agent.ppo_net.critic(obsers)) ** 2)
            value_grads = tape.gradient(value_loss, agent.ppo_net.critic.trainable_variables)
            agent.ppo_net.value_opt.apply_gradients(zip(value_grads, agent.ppo_net.critic.trainable_variables))

    last100Scores[last100ScoresIndex] = episode_length
    last100ScoresIndex += 1
    if last100ScoresIndex >= 50:
        last100Filled = True
        last100ScoresIndex = 0

    if not last100Filled:
        print ("EP " + str(epoch) + " - " + format(episode_length) + "/" + str(max_steps) + " Episode steps " )
    else :    
      m, s = divmod(int(time.time() - start_time), 60)
      h, m = divmod(m, 60)
      print ("EP " + str(epoch) + " - " + format(episode_length) +"/"+str(max_steps) + " Episode steps - last 50 episodes : " + str((sum(last100Scores)/len(last100Scores))) + " - Cumulated R: " + str(episode_return) + "     Time: %d:%02d:%02d" % (h, m, s))
      print ("Success rate: " + str(round(float(number_success)/float(epoch),3)))
      # if (epoch)%100==0:
      #     env._flush()


    if (epoch+1) % 50 == 0:
      plotter.plot(env)
      if learning:
        agent.ppo_net.actor.save('/home/bui1720/gym-gazebo/src/dzung1720/models/1_output/ppoper_actor_512512512_1out_3dyn_ran.h5')
        agent.ppo_net.critic.save('/home/bui1720/gym-gazebo/src/dzung1720/models/1_output/ppoper_critic_512512512_1out_3dyn_ran.h5')
        print("Save the model successfully")

  env.close()

