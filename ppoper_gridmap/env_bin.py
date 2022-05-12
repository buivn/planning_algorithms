import numpy as np
import random

class create_environment_binary:
  color = ['green','blue','yellow','orange','violet', 'pink']

  def __init__(self, obs_num, agent_num, map_dims, n_input, n_bit):
    self.nrows = map_dims
    self.ncolums = map_dims
    self.state_vector_size = n_input*2*n_bit
    self.obs_num = obs_num
    self.agent_num = agent_num
    self.n_input = n_input
    self.n_bit = n_bit
    self.obstacles = []
    self.create_obstacles()
    self.destination = self.set_end_state()
    self.start = self.set_start_state()
    self.state = np.copy(self.start)
    # self.action_space = ['left', 'right', 'up', 'down']
    self.action_space = [0, 1, 2, 3]

  
  def create_obstacles(self):
    # create and draw obstacles
    state_list = np.array([[17, 17], [5, 22], [4, 20], [20, 16], [11, 10], [18, 9], [10, 8], [10, 14], [17, 5], [21, 16], [6, 23], 
                           [13, 19], [5, 19], [14, 17], [12, 21], [23, 14], [9, 21], [23, 13], [20, 7], [20, 6]])
    # state_list = np.array([[4, 16], [7, 15], [14, 10], [14, 14], [8, 7], [5, 7], [10, 6], 
    #                           [11, 12], [10, 3], [5, 8], [12, 14], [3, 12]])  # obstacl3 = 12
    for i in range(self.obs_num):
      while True:
        # randomly picking up obstacles which are different than the target
        obs = np.zeros(2).astype(int)
        obs[0] = state_list[i,0]
        obs[1] = state_list[i,1]
        # obs[0] = random.randint(4,self.ncolums-5)
        # obs[1] = random.randint(4,self.ncolums-5)
        # be sure the obstacle is not overlapped each other
        overlap_obs = False
        for j in range(len(self.obstacles)):
          if (obs[0] == self.obstacles[j][0]) and (obs[1] == self.obstacles[j][1]):
            overlap_obs = True
        if not overlap_obs:
          break     
      self.obstacles.append((obs[0], obs[1]))   
  
  def check_overlap(self, pos):
    overlap_target = False
    # check if the obstacle is too close to the target
    if (abs(pos[0]-self.destination[0]) + abs(pos[1] -self.destination[1])) <=1:
      overlap_target = True

    # be sure it is not overlap the obstacles
    overlap_obs = False
    for j in range(len(self.obstacles)):
      if (pos[0] == self.obstacles[j][0]) and (pos[1] == self.obstacles[j][1]):
        overlap_obs = True
    return (not overlap_target) and (not overlap_obs)
    
  def set_start_state(self):
    start_state = np.zeros(2).astype(int)
    while True:
      # randomly picking up a start state which is not overlapped the target and obstacles
      start_state[0] = random.randint(0,self.ncolums-1)
      start_state[1] = random.randint(0,self.nrows-1)
       # check two overlap condition
      if self.check_overlap(start_state):
        break  
    return start_state

  def set_end_state(self):
    end_state = np.zeros(2).astype(int)
    while True:
      # randomly picking up a start state which is not overlapped the target and obstacles
      end_state[0] = random.randint(0,self.ncolums-1)
      end_state[1] = random.randint(0,self.nrows-1)
      # check obstacle overlap condition
      overlap_obs = False
      for j in range(len(self.obstacles)):
        if (abs(end_state[0]-self.obstacles[j][0]) + abs(end_state[1]-self.obstacles[j][1])) <=1:
          overlap_obs = True
      if not overlap_obs:
        break
    # print('the new end state: ', end_state) 
    return end_state
  
  # convert a decimal position to a binary vector 
  def decToBin(self, num, num_bit):
    vector = np.zeros(num_bit)
    i = num_bit -1
    while num > 1:
      vector[i] = num % 2
      num = num // 2
      i -= 1
    vector[i] = num
    return vector
  
  # convert the decimal input vector into a binary input vector
  def decInputToBinInput(self):
    state_convert = np.zeros([1, self.state_vector_size])
    for i in range(self.n_input):
      # for the current position
      if i == 0:
        binVec_x = self.decToBin(int(self.state[0]), self.n_bit)
        binVec_y = self.decToBin(int(self.state[1]), self.n_bit)    

      # for the target
      if i == 1:    
        binVec_x = self.decToBin(int(self.destination[0]), self.n_bit)
        binVec_y = self.decToBin(int(self.destination[1]), self.n_bit)
      # for the starting position
      if i == 2:
        binVec_x = self.decToBin(int(self.start[0]), self.n_bit)
        binVec_y = self.decToBin(int(self.start[1]), self.n_bit)          
      # save all into a vector
      state_convert[0,2*i*self.n_bit:(2*i+1)*self.n_bit] = binVec_x
      state_convert[0,(2*i+1)*self.n_bit:(2*i+2)*self.n_bit] = binVec_y
    
    return state_convert

  # move to the next state
  def step(self, action):
    state = self.state
    reward = 0.0
    done = False
    if action in self.action_space:
      if action == 0: #'left':
        if state[0] == 0:
          reward = -100.0
          done = True
          # print('this action makes agent get out of gridmap')
        else:
          self.state[0] = state[0] - 1
          if (self.state[0],self.state[1]) in self.obstacles: # collide obstacles -> death
            done = True
            reward = -100.0
          elif (self.state[0] == self.destination[0]) and (self.state[1] == self.destination[1]):
            done = True
            reward = 100.0
          else:
            reward = self.repulsive_field(160, self.state, self.obstacles)

      if action == 1: #'right':
        if state[0] >=  self.ncolums-1:
          reward = -100.0
          done = True
        else:
          self.state[0] = state[0] + 1
          if (self.state[0],self.state[1]) in self.obstacles: # collide obstacles -> death
            done = True
            reward = -100.0
          # reach the target
          elif (self.state[0] == self.destination[0]) and (self.state[1] == self.destination[1]):
            done = True
            reward = 100.0
          else:
            reward = self.repulsive_field(160, self.state, self.obstacles)        

      if action == 2: #'up':
        if state[1] <= 0: # the agent is at the edge of the map
          reward = -100.0
          done = True
        else:
          self.state[1] = state[1] - 1
          if (self.state[0],self.state[1]) in self.obstacles: # collide obstacles -> death
            done = True
            reward = -100.0
          elif (self.state[0] == self.destination[0]) and (self.state[1] == self.destination[1]):
            done = True
            reward = 100.0
          else:
            reward = self.repulsive_field(160, self.state, self.obstacles)        

      if action == 3: #'down':
        if state[1] >= self.nrows-1: # the agent is at the bottom edge of the map
          reward = -100.0
          done = True
        else:
          self.state[1] = state[1] + 1
          if (self.state[0],self.state[1]) in self.obstacles: # collide obstacles -> death
            done = True
            reward = -100.0
          elif (self.state[0] == self.destination[0]) and (self.state[1] == self.destination[1]):
            done = True
            reward = 100.0
          else:
            reward = self.repulsive_field(160, self.state, self.obstacles)
    state_convert = self.decInputToBinInput()
    return state_convert, reward, done, {}
  

  def repulsive_field(self, nuy, state, obstacles):
    mindis_to_obs = 0
    reward = 0.0
    for i in range(len(obstacles)):
      dis = abs(state[0] - obstacles[0][0]) + abs(state[1] - obstacles[0][1]) 
      if i == 0:
        mindis_to_obs = dis
      else:
        if mindis_to_obs > dis:
          mindis_to_obs == dis
    # the reward is calculated by potential field function
    if (mindis_to_obs >= 2) or (len(obstacles)==0):
      reward = -1.0
    else:
      reward = -0.5*nuy*(1/float(mindis_to_obs) - 0.5)**2 # potential field
    return reward

  # this function reset as the agent complete the training for one pair start - end points 
  def reset(self):
    self.destination = self.set_end_state()
    self.start = self.set_start_state()
    self.state = np.copy(self.start)
    # convert the state into a vector
    state_convert = self.decInputToBinInput()
    return state_convert
  
  # this function restart as the agent reach the target 
  def restart(self):
    self.start = self.set_start_state()
    self.state = np.copy(self.start)
    # convert the state into a vector
    state_convert = self.decInputToBinInput()
    return state_convert

  def restart_2(self):
    # self.start = self.set_start_state()
    self.state = np.copy(self.start)
    state_convert = self.decInputToBinInput()
    return state_convert