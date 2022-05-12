#!/bin/bash

# add dynamic and static obstacles env with rectangle and cylinder shapes
if [ -z "$STATIC_DYN_OBSTACLES_REC_CYL_V0" ]; then
  bash -c 'echo "export STATIC_DYN_OBSTACLES_REC_CYL_V0="`pwd`/../assets/worlds/static_dyn_obstacles_rec_cyl_v0.world >> ~/.bashrc'
else
  bash -c 'sed "s,STATIC_DYN_OBSTACLES_REC_CYL_V0=[^;]*,'STATIC_DYN_OBSTACLES_REC_CYL_V0=`pwd`/../assets/worlds/static_dyn_obstacles_rec_cyl_v0.world'," -i ~/.bashrc'
fi

# set the gazebo plugin path for dynamic obstacles
bash -c 'echo "export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:"/home/bui1720/gazebo_animatedbox_tutorial/build >> ~/.bashrc'

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

#Load turtlebot variables. Temporal solution
chmod +x catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em
bash catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em

#add turtlebot launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_MAZE" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_MAZE="`pwd`/../assets/worlds/b_maze_1.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_MAZE=[^;]*,'GYM_GAZEBO_WORLD_MAZE=`pwd`/../assets/worlds/b_maze_1.world'," -i ~/.bashrc'
fi

#add static obstacles v0 - launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V0" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V0="`pwd`/../assets/worlds/static_obstacles_v0.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V0=[^;]*,'GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V0=`pwd`/../assets/worlds/static_obstacles_v0.world'," -i ~/.bashrc'
fi

#add static obstacles v1 - launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V1" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V1="`pwd`/../assets/worlds/static_obstacles_v0.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V1=[^;]*,'GYM_GAZEBO_WORLD_STATIC_OBSTACLES_V1=`pwd`/../assets/worlds/static_obstacles_v0.world'," -i ~/.bashrc'
fi

#add dynamic obstacles v0 - launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_DYNAMIC_OBSTACLES_V0" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_DYNAMIC_OBSTACLES_V0="`pwd`/../assets/worlds/dynamic_obstacles_v0.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_DYNAMIC_OBSTACLES_V0=[^;]*,'GYM_GAZEBO_WORLD_DYNAMIC_OBSTACLES_V0=`pwd`/../assets/worlds/dynamic_obstacles_v0.world'," -i ~/.bashrc'
fi


#copy altered urdf model
cp -r ../assets/urdf/kobuki_nn_urdf/urdf/ catkin_ws/src/kobuki/kobuki_description

#copy laser mesh file
cp ../assets/meshes/lidar_lite_v2_withRay.dae catkin_ws/src/kobuki/kobuki_description/meshes

exec bash # reload bash

