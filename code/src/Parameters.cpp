/*
WHISKiT Physics Simulator
Copyright (C) 2019 Nadina Zweifel (SeNSE Lab)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#include "Parameters.h"

// set default parameter values
Parameters::Parameters(){
	// input arguments for simulator
	DEBUG = 0;			// enable debug mode
	TIME_STEP = 0.001;	// set time step, this is related to output video's FPS
	NUM_STEP_INT = 100;	// set internal time step
	TIME_STOP = 10.0;	// set overall simulation time
    PRINT = 2;			// set to PRINT=1 to kinematics/dynamics realtime, set to PRINT = 2 to print simulation time
	SAVE = 1;			// save results to csv file
	SAVE_VIDEO = 1;		// save video when running main_opengl.cpp
	SAVE_KINEMATICS = 0;// save kinematics to csv file
	CONNECT = 0;		// enable connection between Python and C++
	SIM_TIME = 0.5;		// total simulation time
	// collision object type
	OBJECT = 0;			// 0: nothing
						// 1: peg
						// 2: wall
						// 3: create object from 3D scan
	OBJ_SCALE = 100;	// scale of object
				
	// parameters for peg (OBJECT = 1)
	PEG_LOC = btVector3(20, 25, 0);
	PEG_SPEED = 10;	
	VEL_MODE = 1;

	// specify whisker configuration parameters
	MODEL_TYPE = 0; // Selects model type: 0 => average rat whisker array; 1 => model rat whisker array from Belli et al. 2018
	WHISKER_NAMES = {"RC0", "RC1", "RB1", "RD1", "LC0", "LC1", "LB1", "LD1"}; // select whiskers to simulate
	// WHISKER_NAMES = {"RA0", "RA1", "RA2", "RA3", "RA4", "RB0", "RB1", "RB2", "RB3", "RB4",
	// 				 "RC0", "RC1", "RC2", "RC3", "RC4", "RC5", "RC6", 
	// 				 "RD0", "RD1", "RD2", "RD3", "RD4", "RD5", "RD6", 
	// 				 "RE1", "RE2", "RE3", "RE4", "RE5", "RE6"};
    	
	BLOW = 1;				// increase whisker diameter for better visualization (will affect dynamics!!)
	NO_CURVATURE = 0;		// disable curvature
	NO_MASS = 0;			// disable mass of bodies
	NO_WHISKERS = 0;		// disable whiskers
	NUM_LINKS = 20;			// set number of links
	RHO_BASE = 1260.0;		// set density at whisker base
	RHO_TIP = 1690.0;		// set density at whisker tip
	E = 5e9;				// set young's modulus (GPa) at whisker base
	ZETA = 0.99;			// set damping coefficient zeta at whisker base

	// enable/disable whisking mode for added whiskers
	// Note: the whisking trajectory is pre-specified by user.
	ACTIVE = 1;				
	file_whisking_init_angle = ACTIVE?"whisking_init_angle.csv":"param_bp_angles.csv";
	file_whisking_angle = "whisking_trajectory.csv";

	// enable/disable exploring mode for rat head
	// Note: the head trajectory is 
	EXPLORING = 0;
	dir_rathead = "../data/object/NewRatHead.obj";
	dir_rathead_trajectory = "../data/rathead_trajectory_sample.csv";


	// rat position/orientation parameters
	RATHEAD_LOC = {0,0,0}; 			// set position of rathead
	RATHEAD_ORIENT = {0,0,0}; 		// set initial heading of rathead

	// Object position/orientation default parameters
	ObjX = 0.0;
	ObjY = 0.0;
	ObjZ = 0.0;

	ObjQx = 0.0;
	ObjQy = 0.0;
	ObjQz = 0.0;
	ObjQw = 1.0;

	ObjYAW = 0.0;
	ObjPITCH = 0.0;
	ObjROLL = 0.0;


	// camera parameters for visualization
	CPOS = btVector3(0, 26, 20);	// set camera pos relative to rathead
	CDIST=50;						// set camera distance
	CPITCH=-89;						// set camera pitch
	CYAW=0;							// set camera yaw

	// input/output file paths
	dir_out = "../output/test";
	file_video = "../output/video_test.mp4";
	file_env = "../data/environment/env_1.obj";	

}

// create a vector with same value
std::vector<float> get_vector(float value, int N){
	std::vector<float> vect;
	for(int i=0; i<N; i++){
		vect.push_back(value);
	}
	return vect;
}

// convert string to float vector - not used I think
std::vector<float> stringToFloatVect(std::vector<std::string> vect_string){
	std::string::size_type sz;
	std::vector<float> vect_float;
	for(int i=0; i<vect_string.size(); i++){
		vect_float.push_back(std::stof(vect_string[i],&sz));
	}
	return vect_float;
}

