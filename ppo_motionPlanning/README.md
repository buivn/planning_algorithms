# Description
**PPO - Motion Planning in Gazebo**\
In this project, we start from a simple environment - only static obstacles with corridor shape.
Then we extend to complicated environment.

Checking the trick: - work well for main_ppo.py 
	- Change the environment in several phases: 
		First step: keep agent as long as possible -> each step will take a positive reward
		Second step: as the agent reach the goal with enough episode, then change immediate reward to minus.


We will test ppo first. In PPO, we start with only single output - that is the rotational velocity.

1. PPO with single output: rotational velocity, the linear one is constant = 0.2m/s with per.
It works very well - just run the file: main_ppo.py and modify the environment.

2. Let PPO with single output run with a dynamic obstacle and a random moving obstacle: it still works ok.
- just run the file: main_ppo.py

2. Extend the PPO with two outputs: linear and rotational velocities.
	There is still problem to run the algorithm. It does not converge well. We are determining the reason.
	The file we are working on: main_ppo_2outs.py
	- make the environment simple --- it works

We demone the results in the following video file:\

<a href="http://www.youtube.com/watch?feature=player_embedded&v=aybtCzTSxU8" target="_blank"><img src="https://github.com/buivn/planning_algorithms/blob/master/ppo_motionPlanning/ppo_gazebo.png" alt="" width="300" height="260" border="10" /></a> 