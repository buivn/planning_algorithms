# Author: Hoang-Dung Bui
# email: hbui20@gmu.edu / bui.hoangdungtn@gmail.com
# Description: This environment is set up to train the RL agent with dynamic obstacles and multiple targets
# The environment state consist the data from laser, distance to the target and agent's velocities.


import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from std_msgs.msg import Empty as empty1

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from gym.utils import seeding

class Dynamic_Obstacles_Env_v1(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "Dynamic_Obstacles_Env_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.reset_odom_pub = rospy.Publisher('/mobile_base/commands/reset_odometry', empty1, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.target = [4.0, -3.2]
        self.obstacles = [[-0.5,-2.3], [-0.5,-1.9], [-0.5,-1.5], [-0.5,-1.1], [-0.5,-0.7], [-0.5,-0.3], \
                [-0.1,-2.3], [0.3,-2.3], [0.7,-2.3], [1.1,-2.3], [1.5,-2.3], [-2.0,2.5], [-1.6,2.5], [-1.2, 2.5], \
                [-0.8,2.5], [-0.3,2.5], [-0.2,2.1], [0.0,1.7], [0.8,0.1], [0.95,-0.2], [1.1,-0.5], [1.5,-0.5], \
                [1.9,-0.5], [2.35,-0.5], [0.2, 1.3], [0.35,1.0], [0.55, 0.7], [0.65, 0.4]]

        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def calculate_observation(self, data, pos):
        min_range = 0.2
        done1 = False  # check for collision
        done2 = False   # check of reaching the target

        # if the robot is too close to any point around, terminate/done=True
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done1 = True
        # distance to the target
        delta_x = self.target[0] - pos.pose.pose.position.x
        delta_y = self.target[1] - pos.pose.pose.position.y
        # reach the target
        if (abs(delta_x) < 0.2) and (abs(delta_y) < 0.2):
            done2 = True

        data1 = list(data.ranges)
        data2 = [delta_x, delta_y]
                       
        return data1, data2, done1, done2

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # block until a service is available
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        max_ang_speed = 0.4
        ang_vel = (action[1]-10)*max_ang_speed*0.1 #from (-0.4 to +0.4)
        
        linear_vel = action[0]*0.1

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        pos = None
        while data is None:
            try:
                # create a new subscription to the topic, receive one message, then unsubscribe
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        while pos is None:
            try:
                # create a new subscription to the topic, receive one message, then unsubscribe
                pos = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except:
                pass
        # block until a service is available
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        data1, data2, done1, done2 = self.calculate_observation(data, pos)
        data3 = [linear_vel, ang_vel]

        if done1:
            reward = -200.0
        elif done2:
            reward = 100.0
        else:
            reward = -0.5
            
        done = done1 or done2

        return np.asarray(data1), np.asarray(data2), np.asarray(data3), reward, done, {}

    def reset_with_same_target(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

        # Reset odometry
        self.reset_odom_pub.publish(empty1())
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
            self.reset_odom_pub.publish(empty1())
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        pos = None
        while pos is None:
            try:
                # create a new subscription to the topic, receive one message, then unsubscribe
                pos = rospy.wait_for_message('/odom', Odometry, timeout=5)
                # ROS_INFO("the current position is: ", pos.pose)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        data1, data2, done1, done2 = self.calculate_observation(data, pos)
        data3 = [0.0, 0.0]
        return np.asarray(data1), np.asarray(data2), np.asarray(data3)

    def select_new_target(self, map_size):
        in_obstacles = True
        
        while in_obstacles:
            in_obstacles = False
            x = map_size*np.random.uniform(-1.0, 1.0)
            y = map_size*np.random.uniform(-1.0, 1.0)
            for ob in self.obstacles:
                if abs(x - ob[0]) < 0.2 and abs(y - ob[1]) < 0.2:
                    in_obstacles = True
        return [x, y]

    def reset_with_new_target(self, map_size=3.0):
        # reset velocities to zero
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

        # reset the target
        self.target = self.select_new_target(map_size)

        # Reset odometry
        self.reset_odom_pub.publish(empty1())
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
            self.reset_odom_pub.publish(empty1())
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        pos = None
        while pos is None:
            try:
                # create a new subscription to the topic, receive one message, then unsubscribe
                pos = rospy.wait_for_message('/odom', Odometry, timeout=5)
                # ROS_INFO("the current position is: ", pos.pose)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        data1, data2, done1, done2 = self.calculate_observation(data, pos)
        data3 = [0.0, 0.0]
        return np.asarray(state), np.asarray(data2), np.asarray(data3)