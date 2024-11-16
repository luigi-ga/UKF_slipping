# AMR23-FP1-UkfSlipping


## Installation

Make sure you have followed the installation instructions in [http://wiki.ros.org/Robots/TIAGo/Tutorials](http://wiki.ros.org/Robots/TIAGo/Tutorials), either installing directly on Ubuntu 20.04 or through Docker. Install catkin_tools, create a catkin workspace and clone this repository in the `src` folder. Make sure you are compiling in *Release* mode by properly setting your catkin workspace:
```bash
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
```
Build your code by running the following command:
```bash
catkin build
```

Alternatively, you can clone directly in the `tiago_public_ws/src` created following the installation instructions.


## Usage

### Starting the simulation and loading the world

To run the Gazebo simulation:
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=<WORLD>     
```
Where `<WORLD>` is one of the worlds in `labrob_gazebo_worlds/worlds`.

For example, run 
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=labrob_empty     
```


### Run the simulation

To run the written module:
```bash
roslaunch tiago_slipping_controller tiago_complete.launch
```