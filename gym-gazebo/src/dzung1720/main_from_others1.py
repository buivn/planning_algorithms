import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import scipy.signal
import time
# import random
from tensorflow.keras.models import load_model
from env_bin import create_environment_binary
from rl_agent import ppo_agent
from per import Buffer_3ver


if __name__ == '__main__':

  # Hyperparameters of the PPO algorithm
  n_pairs_layer1 = 600
  epochs_layer1 = 40

  batch_size=300
  mem_size = 210000
  # clip_ratio = 0.2
  # policy_lr = 3e-4
  # value_function_lr = 1e-3
  train_iterations = 3
  lam = 0.97
  target_kl = 0.01
  hidden_sizes = (512, 1024,1024, 512)
  actions = [0,1,2,3]
  map_dimension = 30
  steps_per_epoch = 3*map_dimension
  obstacle_number = 20 #12 #20
  agent_number = 1        # define the number of agent
  n_input = 2 # (1 = current state), (3 = current state, target, and start), (2 = current state, target)
  n_bits = 6
  # observation_dimensions = 2*map_dimension*n_input
  observation_dims = 2*n_input*n_bits
  threshold_core = 100-2*map_dimension

  env = create_environment_binary(obstacle_number, agent_number, map_dimension, n_input, n_bits)

  # Initialize the buffer
  buffer2 = Buffer_3ver(observation_dims, mem_size)

  agent = ppo_agent(observation_dims, actions, hidden_sizes)
  agent.ppo_net.actor = load_model('/home/dzungbui/learning_ws/rl/src/research/ppoper_gridmap/models/30_ppo_per_actor_3600_20o.h5')
  agent.ppo_net.critic = load_model('/home/dzungbui/learning_ws/rl/src/research/ppoper_gridmap/models/30_ppo_per_critic_3600_20o.h5')
  print(agent.ppo_net.actor.summary())
  eps_min = 0.01
  eps_dec = 5e-6
  epsilon = 1.0 
  for pair in range(n_pairs_layer1):
    # buffer2.buffer_reset()
    epsilon = 1.0
    episode_return = 0
    episode_length = 0
    observation, episode_return, episode_length = env.reset(), 0, 0
    # print("A new pairs of start and target")
    # print("The current point: ", env.state)
    print("The target: ", env.destination)
    # print("The state: ", observation)
    score_history = []

    # Iterate over the number of epochs
    for epoch in range(epochs_layer1):
      # Initialize the sum of the returns, lengths and number of episodes for each epoch
      num_episodes = 0
      sum_return = 0
      sum_length = 0  
      # Iterate until it reached target or a max step
      while True:
        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action = agent.select_action(observation)
        observation_new, reward, done, _ = env.step(action[0].numpy())
        # observation_new, reward, done, _ = env.step(action)
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = agent.ppo_net.critic(observation)
        log_prob_t = agent.log_probs(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer2.store_experience(observation, action, reward, value_t, log_prob_t)
        epsilon = epsilon - eps_dec \
                      if epsilon > eps_min else eps_min
        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (episode_length == steps_per_epoch):
            last_value = 0 if done else agent.ppo_net.critic(observation.reshape(1, -1))
            buffer2.finish_trajectory(last_value)
            sum_return = episode_return
            sum_length = episode_length
            num_episodes += 1
            # env.restart_2() -> the same starting point
            # env.restart() -> a new starting point
            observation, episode_return, episode_length = env.restart(), 0, 0
            break
      
      for _ in range(26):
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
            # print(abs_err)
            buffer2.batch_update(tree_idx, abs_err)
            # train the actor
            policy_grads = tape.gradient(policy_loss, agent.ppo_net.actor.trainable_variables)
            agent.ppo_net.policy_opt.apply_gradients(zip(policy_grads, agent.ppo_net.actor.trainable_variables))

            # Train the value function by regression on mean-squared error
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                value_loss = tf.reduce_mean((returns - agent.ppo_net.critic(obsers)) ** 2)
            value_grads = tape.gradient(value_loss, agent.ppo_net.critic.trainable_variables)
            agent.ppo_net.value_opt.apply_gradients(zip(value_grads, agent.ppo_net.critic.trainable_variables))

              # agent.train(obsers, actions, log_probs_b, advantages, returns)

      score_history.append(sum_return)

      print(f" Pair {pair+1} - epoch {epoch + 1} - return value: %6.0f - number of step: %4.0f" % (sum_return, sum_length))
      avg_score = np.mean(score_history[max(0, epoch-20):(epoch+1)])
      if (avg_score > threshold_core) and (epoch > 10):
        break

    if ((pair+1) %50 == 0) and (pair > 10):
      agent.ppo_net.actor.save('/home/dzungbui/learning_ws/rl/src/research/ppoper/models/30_ppo_per_actor_4400_20o.h5')
      agent.ppo_net.critic.save('/home/dzungbui/learning_ws/rl/src/research/ppoper/models/30_ppo_per_critic_4400_20o.h5')