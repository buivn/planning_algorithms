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
from ppo import ppo_agent1
# from per import Buffer_3ver
from per_v1 import Buffer_nonPer
from per_v1 import Buffer_v0

if __name__ == '__main__':
  # tf.compat.v1.disable_eager_execution()
  tf.enable_eager_execution()
  # --------------------------INITIALIZE ALL VALUES OF:...............................#
  agent_number = 1        # define the number of agent
  outdir = '/tmp/gazebo_gym_experiments1/'
  env = gym.make('Static_Obstacles-v0')
  plotter = liveplot.LivePlot(outdir)
  continue_execution = False
  # Hyperparameters of the PPO algorithm
  if not continue_execution:
    #Each time we take a sample and update our weights it is called a mini-batch.
    #Each time we run through the entire dataset, it's called an epoch.
    #PARAMETER LIST
    epochs = 20
    max_steps = 80
    batch_size = 64
    learnStart = 64
    learningRate = 0.0002
    discountFactor = 0.99
    mem_size = 150000
    network_inputs = 100
    n_outputs = 2
    out1_dim = 5
    out2_dim = 21
    network_outputs = 21
    network_structure = [512, 512, 512]
    current_epoch = 0
    train_iterations = 4

  # Initialize the buffer
  # buffer2 = Buffer_3ver(network_inputs, mem_size)
  # buffer2 = Buffer_nonPer(state_dims=network_inputs, action_dims=2, max_size=mem_size)
  buffer2 = Buffer_v0(state_dims=network_inputs, n_outputs=2, max_size=mem_size)

  agent = ppo_agent1(network_inputs, network_outputs, network_structure)
  agent.ppo_net.linear = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_linear_512x3_2outs.h5')
  agent.ppo_net.rot = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_rot_512x3_2outs.h5')
  agent.ppo_net.critic = load_model('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_critic_512x3_2outs.h5')
  agent.ppo_net.critic.summary()
  agent.ppo_net.linear.summary()
  agent.ppo_net.rot.summary()

  
  env._max_episode_steps = max_steps # env returns done after _max_episode_steps
  env = gym.wrappers.Monitor(env, outdir, force=not continue_execution, resume=continue_execution)
  start_time = time.time()

  
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
      # logits, action = agent.select_action(observation)
      logits_linear, logits_rot, action_linear, action_rot = agent.select_action(observation)
      # act = action[0].numpy()
      act1 = action_linear.numpy()
      act2 = action_rot.numpy()
      log_prob_t = agent.log_probs(logits_linear, logits_rot, act1[0], act2[0])
      

      # combine two action into a vector
      act = np.concatenate([act1, act2])
      # print(act)
      observation_new, reward, done, _ = env.step(act)
      episode_return += reward
      episode_length += 1

      # Get the value and log-probability of the action
      value_t = agent.ppo_net.critic(observation)
      value_t = value_t.numpy()    

      # Store obs, act, rew, v_t, logp_pi_t
      buffer2.store_experience(observation, act, reward, value_t, log_prob_t)
      # Update the observation
      observation = observation_new.copy()

      # Finish trajectory if reached to a terminal state
      if done: # or (episode_length >= max_steps):
        last_value = 0 if done else agent.ppo_net.critic(observation.reshape(1, -1)).numpy().squeeze()
        buffer2.finish_trajectory(last_value)
        observation = env.reset()
        if episode_return > 5.0:
          number_success += 1.0
        break
    
    if learning:  # learning - training
      if 0.6 > float(number_success)/float(epoch+1) > 0.3 and epoch > 40:
        num_batch = 10
      elif float(number_success)/float(epoch+1) >= 0.6 and epoch > 40:
        num_batch = 20
      else:
        num_batch = 10
      for _ in range(num_batch):
        if buffer2.pointer > batch_size:
          # Get values from the buffer
          tree_idx, obsers, actions, advantages, returns, log_probs_b = buffer2.get_sample(batch_size)
                  
          # obsers, actions, advantages, returns, log_probs_b = buffer2.get_sample(batch_size)

          for _ in range(train_iterations):
            with tf.GradientTape(persistent=True) as tape:  # Record operations for automatic differentiation.
              linear = agent.ppo_net.linear(obsers)
              rot = agent.ppo_net.rot(obsers)
              # print(linear.numpy())
              log_probs = agent.log_probs(linear, rot, actions[:,0], actions[:,1])

              diff1 = log_probs[0] - log_probs_b[:,0]
              diff2 = log_probs[1] - log_probs_b[:,1]
              
              ratio1 = tf.exp(diff1)
              ratio2 = tf.exp(diff2)
              min_advantage = tf.where(
                  advantages > 0,
                  (1 + agent.clip_ratio) * advantages,
                  (1 - agent.clip_ratio) * advantages, )
        
              # clipped_probs_1 = tf.clip_by_value(
              #                           ratio1,
              #                           (1 + agent.clip_ratio),
              #                           (1 - agent.clip_ratio))              
              # min_advantage1 = clipped_probs_1 * advantages


              # clipped_probs_2 = tf.clip_by_value(
              #                           ratio2,
              #                           (1 + agent.clip_ratio),
              #                           (1 - agent.clip_ratio))              
              # min_advantage2 = clipped_probs_2 * advantages
              
              policy_loss1 = -tf.reduce_mean(tf.minimum(ratio1 * advantages, min_advantage))
              policy_loss2 = -tf.reduce_mean(tf.minimum(ratio2 * advantages, min_advantage))

            # print("Is this still ok here-------------------")
              
            # update the priorities of the data
            abs_err = np.abs(diff1.numpy().squeeze()) + np.abs(diff2.numpy().squeeze())

            buffer2.batch_update(tree_idx, abs_err)
            # train the actor
            policy_grad1 = tape.gradient(policy_loss1, agent.ppo_net.linear.trainable_variables)
            agent.ppo_net.policy_opt.apply_gradients(zip(policy_grad1, agent.ppo_net.linear.trainable_variables))

            policy_grad2 = tape.gradient(policy_loss2, agent.ppo_net.rot.trainable_variables)
            agent.ppo_net.policy_opt.apply_gradients(zip(policy_grad2, agent.ppo_net.rot.trainable_variables))

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
      print ("EP " + str(epoch) + " - " + format(episode_length) + "/" + str(max_steps) + " Episode steps - last 50 Steps : " + str((sum(last100Scores)*2 / len(last100Scores))) + " - Cumulated R: " + str(episode_return) + "     Time: %d:%02d:%02d" % (h, m, s))
      print ("Success rate: " + str(round(float(number_success)/float(epoch),3)))
      # if (epoch)%100==0:
      #     env._flush()

    if (epoch+1) % 50 == 0:
        plotter.plot(env)
        if learning:
          agent.ppo_net.linear.save('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_linear_512x3_2outs.h5')
          agent.ppo_net.rot.save('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_rot_512x3_2outs.h5')
          agent.ppo_net.critic.save('/home/bui1720/gym-gazebo/src/dzung1720/models/ppoper_critic_512x3_2outs.h5')
          print("Save the model successfully")

  env.close()

