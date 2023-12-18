# Author: Hoang-Dung Bui
# email: hbui20@gmu.edu / bui.hoangdungtn@gmail.com
# Description: This environment is set up to run the potential field algorithm.
# The environment state consist the data from laser, distance to the target and agent's velocities.


# import gym
import rospy
import roslaunch
import time
import numpy as np

from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from std_msgs.msg import Empty as empty1
from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from math import atan2, sqrt
# import math


class obstacles_env_v0(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "Env_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)

        self.reset_odom_pub = rospy.Publisher('/mobile_base/commands/reset_odometry', empty1, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        # self.target = [5.5, -0.4]
        # self.target = [5.3, -2.0]
        # self.target = [5.0, -3.0]
        # self.target = [4.5, -3.5]
        self.target = [3.0, -5.0]
        
        # self.target = [0.8, -5.0]

        self.reward_range = (-np.inf, np.inf)
        rospy.rate = rospy.Rate(20)

    def potential_function(self, data, pos):
        vel = 0.0
        rot = 0.0
        reach_goal = False
        limit_speed = False

        # distance to the target
        distance_x = self.target[0] - pos.pose.pose.position.x
        distance_y = self.target[1] - pos.pose.pose.position.y

        rot_q = pos.pose.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        # reach the target
        if (abs(distance_x) < 0.15) and (abs(distance_y) < 0.15):
            reach_goal = True
            vel = 0.0
            rot = 0.0
            # return 0, 0, reach_goal
        else:
            # if the robot is too close to any point around, terminate/done=True
            laser_data = np.array(data.ranges)
            laser_data[np.where(laser_data==float('inf'))] = 20.0
            
            index = np.argmin(laser_data)
            r_force = self.repulsive_potential(laser_data[index])
            angle_to_ob = theta + (index-50.0)*0.0476
            current_pos = [pos.pose.pose.position.x, pos.pose.pose.position.y]
            a_force, angle_to_goal, dis_to_goal = self.attracttive_potential(current_pos, self.target)


            if r_force == 0.0 or (laser_data[index] - dis_to_goal) > 0.3 or \
                        (laser_data[index] > 1.6 and abs(angle_to_goal - angle_to_ob) > 0.7):
                next_angle = angle_to_goal
            elif laser_data[index] <= 1.6:
                if abs(angle_to_goal - angle_to_ob) <= 0.6:
                    if angle_to_goal > angle_to_ob:
                        vel = 0.25
                        rot = 0.31
                    else:
                        vel = 0.25
                        rot = -0.31
                    return vel, rot, reach_goal
                else:
                    next_angle = angle_to_goal
                    limit_speed = True

            else:
                next_angle = angle_to_goal
                limit_speed = True

            # angle_to_goal = atan2 (distance_y, distance_x)
            if abs(next_angle - theta) >0.1:
                vel = 0.02
                if next_angle - theta > 0:
                    rot = 0.15
                else:
                    rot = -0.15 
            elif limit_speed:
                vel = 0.1
                rot = 0.0
            else:
                if a_force > 1.0:
                    vel = 0.3
                else:
                    vel = 0.15
                rot = 0.0

        return vel, rot, reach_goal
        # return 0.0, 0.0, reach_goal

    def repulsive_potential(self, obst, dstar_obst=2.0,nuy=1.0):
        r_f = 0.0
        if obst > dstar_obst:
            r_f = 0.0
        else:
            r_f = 0.5*nuy*pow((1/(obst*0.1) - 1/(0.1*dstar_obst)),2)
        return r_f

    def attracttive_potential(self, current_pos, goal, dstar=2.0, factor=0.3):
        distance = sqrt((current_pos[1]-goal[1])**2 + (current_pos[0]-goal[0])**2)
        a_f = 0.0
        if distance > dstar:
            a_f = dstar*factor*(distance - 0.5*dstar)
        else:
            a_f = 0.5*factor*distance*distance
        angle_to_goal1 = atan2 (goal[1]-current_pos[1], goal[0]-current_pos[0])
        return a_f, angle_to_goal1, distance


    def step(self):
        # block until a service is available
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

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

        # print("Information of the laser: ")

        linear_vel, ang_vel, done = self.potential_function(data, pos)


        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)


        return done


if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    # env = gym.make('Static_Obstacles-v0')
    env = obstacles_env_v0()
    done = False
    while not done:
        done = env.step()
        if done:
            print("Reach the goal")

