import numpy as np
import scipy.signal
import time
import random

class Buffer_nonPer:  # stored as ( state, action, reward, next_state ) in SumTree
    # Buffer for storing trajectories
    def __init__(self, state_dims, action_dims, max_size, gamma=0.99, lam=0.95):

      self.observations = np.zeros(
          (max_size, state_dims), dtype=np.float32)
      self.actions = np.zeros((max_size, action_dims), dtype=np.int32)
      self.advantages = np.zeros(max_size, dtype=np.float32)
      self.rewards = np.zeros(max_size, dtype=np.float32)
      self.returns = np.zeros(max_size, dtype=np.float32)
      self.values = np.zeros(max_size, dtype=np.float32)
      self.log_probs = np.zeros((max_size, action_dims), dtype=np.float32)
      self.pointer = 0
      self.trajectory_start_index = 0
      self.gamma, self.lam = gamma, lam
      self.max_size = max_size
      self.memory_full = False
      self.current_size = 0


    # Each new experience will have a score of max_prority.
    def store_experience(self, observation, action, reward, value, log_prob):
      # save the data
      self.observations[self.pointer] = observation[0]
      self.actions[self.pointer] = action
      self.rewards[self.pointer] = reward
      self.values[self.pointer] = value
      self.log_probs[self.pointer,0] = log_prob[0]
      self.log_probs[self.pointer,1] = log_prob[1]
      self.pointer += 1


    def finish_trajectory(self, last_value=0):
      # Finish the trajectory by computing advantage estimates and rewards-to-go
      path_slice = slice(self.trajectory_start_index, self.pointer) # get the list of number from index1+1 to index2
      rewards = np.append(self.rewards[path_slice], last_value)
      values = np.append(self.values[path_slice], last_value)

      deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
      # values[:-1] from the first item to the item before the last
      # values[1:] from the second item to the last
      # update the advantages and returns values
      self.advantages[path_slice] = self.discounted_cumulative_sums(
          deltas, self.gamma * self.lam )
      self.returns[path_slice] = self.discounted_cumulative_sums(
          rewards, self.gamma)[:-1]

      self.trajectory_start_index = self.pointer

    def discounted_cumulative_sums(self, x, discount):
      # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
      return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]       
    
    # get a batch_size number randomly in range [0, priority_total]
    # search and retrieve in the sumtree the indices corresponding to priority score
    def get_sample(self, batch_size):
      if self.memory_full:
          max_mem = self.current_size
      else:
          max_mem = self.pointer
      batch = np.random.choice(max_mem, batch_size)
      obsers = self.observations[batch]
      actions = self.actions[batch]
      advans = self.advantages[batch]
      returns = self.returns[batch]
      log_probs = self.log_probs[batch]


      # reset the buffer if it is nearly full
      if self.pointer > (self.max_size - 500):
        self.trajectory_start_index = 0
        self.pointer = 0
        self.current_size = self.pointer
        self.memory_full = True
      return  obsers, actions, advans, returns, log_probs
    




class SumTree(object):
  # Here we initialize the tree with all nodes = 0
  def __init__(self, capacity):
    # Number of leaf nodes (final nodes) that contains experiences
    self.capacity = capacity
    # Generate the tree with all nodes values = 0
    self.tree = np.zeros(2 * capacity - 1)
  
  # add our priority score in the sumtree leaf
  def add(self, data_idx, priority):
    # Look at what index we want to put the experience
    tree_index = data_idx + self.capacity - 1
    # Update the leaf
    self.update (tree_index, priority)
          
  # Update the leaf priority score and propagate the change through tree
  def update(self, tree_index, priority):
    # Change = new priority score - former priority score
    change = priority - self.tree[tree_index]
    self.tree[tree_index] = priority
    # propagate the change through the tree
    while tree_index != 0:
        tree_index = (tree_index - 1) // 2
        self.tree[tree_index] += change
      
  # get a leaf from the tree and return  leaf_idx, priority, and data_index
  def get_leaf(self, v): # v is a priority
    parent_idx = 0
    while True:
      l_child_idx = 2 * parent_idx + 1
      r_child_idx = l_child_idx + 1
      # If reaching bottom, end the search
      if l_child_idx >= len(self.tree):
        leaf_idx = parent_idx
        break
      else: # downward search, always search for a higher priority node
        if v <= self.tree[l_child_idx]:
          parent_idx = l_child_idx
        else:
          v -= self.tree[l_child_idx]
          parent_idx = r_child_idx
    return leaf_idx, self.tree[leaf_idx]

  @property
  def total_priority(self):
    return self.tree[0] # Returns the root node


# Now we finished constructing our SumTree object, next we'll build a memory object.
class Buffer_3ver:  # stored as ( state, action, reward, next_state ) in SumTree
    # Buffer for storing trajectories
    def __init__(self, state_dims, max_size, gamma=0.99, lam=0.95):
      # A tree for data priority
      self.priority_tree = SumTree(max_size)
      self.observations = np.zeros(
          (max_size, state_dims), dtype=np.float32)
      # # print(self.observation_buffer.shape)
      self.actions = np.zeros(max_size, dtype=np.int32)
      self.advantages = np.zeros(max_size, dtype=np.float32)
      self.rewards = np.zeros(max_size, dtype=np.float32)
      self.returns = np.zeros(max_size, dtype=np.float32)
      self.values = np.zeros(max_size, dtype=np.float32)
      self.log_probs = np.zeros(max_size, dtype=np.float32)
      self.pointer = 0
      self.trajectory_start_index = 0
      self.gamma, self.lam = gamma, lam
      self.max_size = max_size
      # self.observation_dimensions = observation_dims
      self.per_e = 0.01  # avoid some experiences to have 0 probability of being taken
      self.per_a = 0.6  # tradeoff between taking only high priority & sampling randomly
      self.per_b = 0.4  # importance-sampling, from initial value increasing to 1
      self.per_b_increment_by_step = 0.001 # reduce the random sampling by time
      self.abs_err_upper = 1.  # clipped abs error


    # Each new experience will have a score of max_prority.
    def store_experience(self, observation, action, reward, value, log_prob):
      # Find the max priority in all the leaf node
      max_priority = np.max(self.priority_tree.tree[-self.priority_tree.capacity:])
      # If max_priority = 0, set it a nonzero value (have chance to be picked)
      if max_priority == 0:
          max_priority = self.abs_err_upper
      self.priority_tree.add(self.pointer, max_priority)   # set the max priority for new experience
      # save the data
      self.observations[self.pointer] = observation[0]
      self.actions[self.pointer] = action
      self.rewards[self.pointer] = reward
      self.values[self.pointer] = value
      self.log_probs[self.pointer] = log_prob
      self.pointer += 1

    def finish_trajectory(self, last_value=0):
      # Finish the trajectory by computing advantage estimates and rewards-to-go
      path_slice = slice(self.trajectory_start_index, self.pointer) # get the list of number from index1+1 to index2
      rewards = np.append(self.rewards[path_slice], last_value)
      values = np.append(self.values[path_slice], last_value)

      deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
      # values[:-1] from the first item to the item before the last
      # values[1:] from the second item to the last
      self.advantages[path_slice] = self.discounted_cumulative_sums(
          deltas, self.gamma * self.lam )
      self.returns[path_slice] = self.discounted_cumulative_sums(
          rewards, self.gamma)[:-1]

      self.trajectory_start_index = self.pointer

    def discounted_cumulative_sums(self, x, discount):
      # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
      return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]       
    
    # get a batch_size number randomly in range [0, priority_total]
    # search and retrieve in the sumtree the indices corresponding to priority score
    def get_sample(self, batch_size):
      i_list = [] 
      leaf_indices = []
      # divide the priority range [0, ptotal] into batch_size ranges
      priority_segment = self.priority_tree.total_priority / batch_size
      for i in range(batch_size):
        # get a value between two thresholds of priorities
        a, b = priority_segment * i, priority_segment * (i + 1)
        value = np.random.uniform(a, b)
        # Experience that correspond to each value is retrieved
        leaf_idx, priority = self.priority_tree.get_leaf(value)
        data_idx = leaf_idx - self.max_size + 1
        i_list.append(data_idx)
        leaf_indices.append(leaf_idx)
      # reset the buffer if it is nearly full
      if self.pointer > (self.max_size - 1200):
        self.trajectory_start_index = 0
        self.pointer = 0
      return leaf_indices, self.observations[i_list], self.actions[i_list], self.advantages[i_list], self.returns[i_list], self.log_probs[i_list]
    
    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.per_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.per_a)

        for ti, p in zip(tree_idx, ps):
            self.priority_tree.update(ti, p)