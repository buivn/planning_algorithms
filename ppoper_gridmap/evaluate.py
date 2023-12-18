from load_ppoper import Agent
import numpy as np
import random
from utils import PlotModel, nn_map_ploting, visualize_evaluation
import tensorflow as tf
import pyglet 
import pylab

def test_RL(agent, obstacles, start, destination, map_dimension, cell_size):
  agent.load()
  title = "ppoper"
  nn_map_ploting(title, 1100, 1100, cell_size, obstacles, start, destination, agent.actor, map_dimension, '50_ppo.ps')

def visualize_results(agent, obstacles, start, destination, map_dimension, cell_size):
  agent.load()
  title = "ppoper"
  visualize_evaluation(title, 650, 650, cell_size, obstacles, start, destination, agent.actor, map_dimension, '50_ppo.ps')


if __name__ == '__main__':
  tf.compat.v1.disable_eager_execution()
  # --------------------------INITIALIZE ALL VALUES OF:...............................#
  # env = create_environment(obstacle_number, agent_number, map_dimension)
  map_dimension = 30
  cell_size=20
  obstacle10 = [(1, 5), (8, 3), (6, 6), (3, 5), (2, 8), (3, 3)]
  obstacle20 = [(4, 16), (7, 15), (14, 10), (14, 14), (8, 7), (5, 7), (10, 6), (11, 12), (10, 3), (5, 8), (12, 14), (3, 12)]
  obstacle30 = [(17, 17), (5, 22), (4, 20), (20, 16), (11, 10), (18, 9), (10, 8), (10, 14), (17, 5), (21, 16), (6, 23), 
                           (13, 19), (5, 19), (14, 17), (12, 21), (23, 14), (9, 21), (23, 13), (20, 7), (20, 6)]
  obstacles = obstacle30

  destination = [26, 3] 
  # [3, 9]
  path_model = './models/30_ppo_per_actor_4400_20o.h5'
  agent = Agent(path=path_model)
  start = [8,8]
  # test_RL(agent, obstacles, start, destination, map_dimension, cell_size)
  visualize_results(agent, obstacles, start, destination, map_dimension, cell_size)
