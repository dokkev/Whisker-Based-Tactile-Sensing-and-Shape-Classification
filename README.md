


<img src="docs/demo.gif">


## Introduction
The objective of this project aims to replicate a rat’s active vibrissal sensing to classify concave and convex objects using artificial neural networks in simulations. Moreover, we investigated how individual whiskers affect neurons in the neural networks of the deep-q learning algorithm, which enabled the rat in simulation to find an optimal whisking orientation that maximizes the symmetry of contacting whiskers.

For detailed information about this project, check out my [post](https://dokkev.github.io/Whisker) from my [portfolio](https://dokkev.github.io)

<img src="docs/whiskit_physics_logo_bg_white.png" height="203px" width="444px" >

I modified this simulator [WHISKiT Physics Simulator](), a 3D dynamical model of the full rat vibrissal array using the open-source physics engine Bullet Physics and OpenGL, for this project.

This repository provides modified WHISKiT Physics Simulator with additional function of
- Parallel Simulation using Northwestern University's Computing Cluster [QuEST](https://www.it.northwestern.edu/research/user-services/quest/)
- Customizable location of 3D modeling
- Real-time commnication node with Python 3
- Some example scripts to filter output
- Tabular data classifers for concave and convex obejct
- Image classifiers for concave and convex object
- DQN script for reinforcement learning


## Installation Instructions:
1. Install OpenGL/Glut with `sudo apt-get install freeglut3-dev`

2. Install Boost 1.62 library with `sudo apt-get install libboost1.62-all-dev`

3. Clone this repository:

```
	git clone https://github.com/dokkev/Whisker-Based-Tactile-Sensing-and-Shape-Classification.git
```

4. Compile whisketphysics:
```
	cd your/path/to/whiskitphysics/code
	mkdir build
	cd build
	cmake ..
	make

```

   If boost library is not found by cmake try:

```
	cd your/path/to/whiskitphysics/code
	mkdir build
	cd build
	sudo cmake --check-system-vars ..
	sudo make

```
5. Run `whiskit` (no graphics) or `whiskit_gui` (with graphics). Use --help or -h for information about command line arguments. Bash scripts for simulation presets are available in "script" folder.

6. If you want to run it with `ROS` copy `whisker_ros` to your catkin workspace and `catkin_make`

### whiskitphysics
This section explains directories in `code`

`config` : some configuration for object tranformation for Quest Simulation is stored here
`data` : contains `.obj` file od 3d modelings and whisker trajectories
`filter_oupt` : contains some python scripts that I used to filter output and generate training input for classifiers
`image_classifer` : binary image classifer and multiclass image classifer with image training input
`include` : header directory for soruce codes
`quest_scripts` : sample scripts for quest simulation
`scripts` : bash scripts with pre-defined parameters for `whiskit_gui` or `whiskit` and python scripts for real-time commuication and DQN
`src` : source code
`tabular_classifer` : binary classifer and multiclass classifer with tabuar training input


#### Simulation with Keyboard Control
run `sh run_user_control` and run `python3 keyboard_control.py` on a sperate terminal

```
	w : move forward
	s : move backward
	a : turn left
	d : turn right
	up-arrow : look up
	down-arrow : look down
```

#### Tabular Classiifer
run `python3 tabular_classifer` or `python3 tabular_multiclass_classifer`

These scripts automaitcally and randomly split test data from train data.
The last column of the data should be classifcation index.

Only data for `tabular_classifer` is provided in this repository

run `python3 logistic regression` for training tabular data with logistic regression method

#### ImageClassiifer
run `python3 classifer` or `python3 multiclass_classifer`

obejcts with differnt classification index should be separated. For example, if my target dir for train is `contact`, this dir should contains two seperate dirs called `concave` and `convex`

#### Reinforcement Learning
run `python3 RL_DQN.py` and `sh run_user_control` in the `script` dir


### whisker_ros

This is ROS package for data visualization and commuication that can be used parallely with WHISKiT Physics Simulator.
This is still under development, but you can run some demos.

run `roslaunch whisker_ros start.launch` 
open up a separate termail and run `sh run_user_control` in the script dir
this launch file will publsish whisker data as `ROS topics`

run `rosrun whisker_ros tabular realtime classifer` to see real-time classification
Make sure that your loaded model input has matching input size with the number of whiskers currently running in the simulation




 

