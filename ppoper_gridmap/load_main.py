from load_ppoper import Agent
import numpy as np
import random
from utils import PlotModel, nn_map_ploting
import tensorflow as tf
import pyglet 
import pylab

def test_RL(agent, obstacles, start, destination, map_dimension, cell_size):
  agent.load()
  # title = "msme-a2c"
  # x = random.randint(0,map_dimension-1)
  # y = random.randint(0,map_dimension-1)
  # start = np.array([x, y])

  title = "ppoper"
  nn_map_ploting(title, 1100, 1100, cell_size, obstacles, start, destination, agent.actor, map_dimension, '50_ppo.ps')


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  # --------------------------INITIALIZE ALL VALUES OF:...............................#
  # env = create_environment(obstacle_number, agent_number, map_dimension)
  map_dimension = 30
  cell_size=20
  # obstacles = [(20, 33), (25, 27), (19, 26), (25, 22), (33, 34), (17, 24), (34, 17), (33, 22), (15, 24), (20, 17), (28, 25), (17, 23), (23, 32), (35, 31), (29, 31), (30, 28), (29, 33), (26, 32), (33, 26), (19, 23), (24, 23), (29, 26), (18, 35), (20, 22), (34, 29), (31, 34), (17, 29), (20, 26), (23, 25), (21, 25), (32, 29), (26, 20), (24, 18), (24, 19), (15, 29), (16, 19), (25, 20), (20, 30), (29, 24), (28, 20)]
  # obstacles = [] #[(4, 3), (2, 2)] #, (10, 12), (8, 3), (4, 8), (9, 11), (10, 10), (12, 8), (12, 3), (3, 12), (8, 7)]
  obstacle10 = [(1, 5), (8, 3), (6, 6), (3, 5), (2, 8), (3, 3)]
  obstacle20 = [(4, 16), (7, 15), (14, 10), (14, 14), (8, 7), (5, 7), (10, 6), (11, 12), (10, 3), (5, 8), (12, 14), (3, 12)]
  obstacle30 = [(17, 17), (5, 22), (4, 20), (20, 16), (11, 10), (18, 9), (10, 8), (10, 14), (17, 5), (21, 16), (6, 23), 
                           (13, 19), (5, 19), (14, 17), (12, 21), (23, 14), (9, 21), (23, 13), (20, 7), (20, 6)]
  obstacles = obstacle30

  destination = [26, 3] 
  # [3, 9]
  path_model = '/home/dzungbui/learning_ws/rl/src/research/ppoper/models/30_ppo_per_actor_4400_20o.h5'
  agent = Agent(path=path_model)
  start = [8,8]
  test_RL(agent, obstacles, start, destination, map_dimension, cell_size)
