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

This code is based on code published by
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2015 Google Inc. http://bulletphysics.org

*/


#include "Simulation.h"

btVector4 BLUE = btVector4(0.,0.,1.0,1);
btVector4 RED = btVector4(1.,0.,0.0,1);
btVector4 GREEN = btVector4(0.,1.,0.0,1);
btVector4 GREY = btVector4(0.,0.,0.0,0.5);
btVector4 YELLOW = btVector4(1.,1.,0.0,1);
btVector4 ORANGE = btVector4(1.,0.647,0.0,1);

void Simulation::stepSimulation(){
	auto start = std::chrono::high_resolution_clock::now(); 
	m_time += parameters->TIME_STEP; 								// increase time
	m_step += 1;													// increase step
	m_time_stop = parameters->SIM_TIME;								// set time stop 
	// std::cout << "Time Stop: " << m_time_stop << " s" << std::endl;
	

	if(parameters->CONNECT){

		scabbers->dump_M(data_dump);
		scabbers->dump_F(data_dump);
		scabbers->dump_Q(data_dump);

		connection.send_data(data_dump);
		// std::cout << "data_dump " << data_dump << std::endl;
		clear_output(data_dump);

		// receive data from Python and update Rat transfomration
		connection.receive_data(state->new_state);
		state->update();
	}

	// run simulation as long as stop time not exceeded
	if(m_time_stop==0 || m_time < m_time_stop){

		// register collisions
		scabbers->detect_collision(m_dynamicsWorld);
		
		// first, push back data into data_dump 
		if(!parameters->NO_WHISKERS && parameters->SAVE){
			scabbers->dump_M(data_dump);
			scabbers->dump_F(data_dump);
			scabbers->dump_Q(data_dump);

			// std::cout << "scabbers: " << scabbers << std::endl;
		}

		// moving object 1
		if(parameters->OBJECT==1){
			if(parameters->PEG_SPEED>0){
				btVector3 velocity = parameters->PEG_SPEED * btVector3(0.4,-1,0).normalized();
				peg->setLinearVelocity(velocity);
				
			}
			
		}

		// move array if in ACTIVE mode
		if(parameters->ACTIVE && !parameters->NO_WHISKERS){
			scabbers->whisk(m_step, parameters->WHISKER_VEL);
			
		}
		
		// move rat head if in EXPLORING mode
		if(parameters->EXPLORING){
			this_loc_vel = parameters->HEAD_LOC_VEL[m_step-1];
			scabbers->setLinearVelocity(btVector3(this_loc_vel[3], this_loc_vel[4], this_loc_vel[5]/10));
			// scabbers->setLinearVelocity(btVector3(0, 0, 0));
			scabbers->setAngularVelocity(btVector3(this_loc_vel[6], this_loc_vel[7], this_loc_vel[8]));
			// scabbers->setAngularVelocity(btVector3(0, 0, 0));
		}
		// Set velocity of the rat with user input
		if (parameters->VEL_MODE){
			

			scabbers->setLinearVelocity(btVector3(0, 0, 0));
			scabbers->setAngularVelocity(btVector3(0, 0, 0));
		}

	

		// Control the rat if CONNECT == 1
		if(parameters->CONNECT){
		// 	scabbers->rotateHead(state->current_state.pitch,state->current_state.yaw,state->current_state.roll);
		// 	scabbers->translateHead(btVector3(state->current_state.x,state->current_state.y,state->current_state.z));
			btQuaternion quat;
			quat.setEulerZYX(state->current_state.yaw,state->current_state.pitch,state->current_state.roll);
			btVector3 vec = btVector3(state->current_state.x,state->current_state.y,state->current_state.z);
			btTransform RatTF = btTransform(quat,vec);
			scabbers->setTransform(RatTF);
		}

		// step simulation
		m_dynamicsWorld->stepSimulation(parameters->TIME_STEP,parameters->NUM_STEP_INT,parameters->TIME_STEP/parameters->NUM_STEP_INT);

		// draw debug if enabled
	    if(parameters->DEBUG){
	    	m_dynamicsWorld->debugDrawWorld();
	    }

	    // set exit flag to zero
	    exitSim = 0;
	
	}
	else{
		// timeout -> set exit flg
		exitSim = 1;
	}
	
	
	auto stop = std::chrono::high_resolution_clock::now(); 
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	m_time_elapsed += duration.count()/1000.f;
	auto factor = m_time_elapsed / m_time;
	auto time_remaining = (int)((m_time_stop - m_time) * (factor));
	if(parameters->PRINT==2){
		std::cout << "\rSimulation time: " << std::setprecision(2) << m_time << "s\tCompleted: " << std::setprecision(2) << m_time/parameters->TIME_STOP*100 << " %\tTime remaining: " << std::setprecision(4) << time_remaining/60 << " min " << std::setprecision(4) << (time_remaining % 60) << " s\n" << std::flush;
	}

	else if(parameters->PRINT==3){
		char do_nothing = 1;

	}
    
}

void Simulation::initPhysics()
{	
	vec = btVector3(0.5,-1,0).normalized();
	data_dump->init(parameters->WHISKER_NAMES);
	

	// set visual axis
	m_guiHelper->setUpAxis(2);

	// create empty dynamics world[0]
	m_collisionConfiguration = new btDefaultCollisionConfiguration(); 
    m_dispatcher = new	btCollisionDispatcher(m_collisionConfiguration); 

    // broadphase algorithm
    m_broadphase = new btDbvtBroadphase();

	// select solver
	std::cout << "Using btSequentialImpulseConstraintSolver..." << std::endl;
	m_solver = new btSequentialImpulseConstraintSolver();

	m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);			

	// set number of iterations
	m_dynamicsWorld->getSolverInfo().m_numIterations = 20;
	m_dynamicsWorld->getSolverInfo().m_solverMode = SOLVER_SIMD |
                        SOLVER_USE_WARMSTARTING |
                        SOLVER_RANDMIZE_ORDER |
                        0;
	m_dynamicsWorld->getSolverInfo().m_splitImpulse = true;
	m_dynamicsWorld->getSolverInfo().m_erp = 0.8f;

	
	// set gravity
	m_dynamicsWorld->setGravity(btVector3(0,0,0));

    // create debug drawer
	m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

	if (m_dynamicsWorld->getDebugDrawer()){
		if(parameters->DEBUG==1){
			std::cout << "DEBUG option 1: wireframes." << std::endl;
			m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawWireframe);
		}
		else if(parameters->DEBUG==2){
			std::cout << "DEBUG option 2: constraints." << std::endl;
			m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawConstraints);
		}
		else if(parameters->DEBUG==3){
			std::cout << "DEBUG option 3: wireframes & constraints." << std::endl;
			m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawWireframe+btIDebugDraw::DBG_DrawConstraintLimits);
		}
		else if(parameters->DEBUG==4){
			std::cout << "DEBUG option 4: AAbb." << std::endl;
			m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawAabb);
		}
		else if(parameters->DEBUG==5){
			std::cout << "DEBUG option 5: Frammes." << std::endl;
			m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawFrames);
		}
		else if(parameters->DEBUG==6){
			std::cout << "DEBUG option 6: Only collision" << std::endl;
			// m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawFrames);
		}
		else{
			std::cout << "No DEBUG." << std::endl;
			m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_NoDebug);
		}
	}

	// add rat to world
	scabbers = new Rat(m_guiHelper,m_dynamicsWorld, &m_collisionShapes, parameters);
	btVector3 rathead_pos = scabbers->getPosition();
	
	// create object to collide with
	// peg
	if(parameters->OBJECT==1){
		btCollisionShape* pegShape = new btCylinderShapeZ(btVector3(1,1,80));
		pegShape->setMargin(0.1);
		m_collisionShapes.push_back(pegShape);
		btTransform trans = createFrame(parameters->PEG_LOC,btVector3(0, 0, 0));
		peg = createDynamicBody(1,0.5,trans, pegShape, m_guiHelper,  BLUE);
		m_dynamicsWorld->addRigidBody(peg,COL_ENV,envCollidesWith);
		peg->setActivationState(DISABLE_DEACTIVATION);
		
	}
	// create object to collide with wall
	else if(parameters->OBJECT==2){
		btCollisionShape* wallShape = new btBoxShape(btVector3(5,200,60));
		wallShape->setMargin(0.1);
		m_collisionShapes.push_back(wallShape);
		btTransform trans = createFrame(btVector3(50,0,0),btVector3(0,0,PI/6));
		wall = createDynamicBody(0,0.5, trans, wallShape, m_guiHelper,  BLUE);
		m_dynamicsWorld->addRigidBody(wall,COL_ENV,envCollidesWith);
	}
	// create object from 3D scan at the origin (0,0,0)
	else if(parameters->OBJECT==3){
		// add environment to world
		btVector4 envColor = btVector4(0.6,0.6,0.6,1);
		env = new Object(m_guiHelper,m_dynamicsWorld, &m_collisionShapes,btTransform(),parameters->file_env,envColor,btScalar(SCALE),btScalar(0),COL_ENV,envCollidesWith);
	}
	
	// create object from 3D scan at a specific position and orientation using 4D quaternion (x,y,z,w)
	else if(parameters->OBJECT==4){
		// add environment to world
		btVector4 envColor = btVector4(0.6,0.6,0.6,1);

		// object position
		ObjX_ = parameters->ObjX;
		ObjY_ = parameters->ObjY;
		ObjZ_ = parameters->ObjZ;

		// object orientation with 4D quaternion	
		ObjQx_ = parameters->ObjQx;
		ObjQy_ = parameters->ObjQy;
		ObjQz_ = parameters->ObjQz;
		ObjQw_ = parameters->ObjQw;

		// make sure params are adjusted accordingly with the user input
		// std::cout << "ObjX: " << ObjX_ << std::endl;

		//set Quaternion and Vector 
		btQuaternion quat = btQuaternion(ObjQx_,ObjQy_,ObjQz_,ObjQw_);
		btVector3 vec = btVector3(ObjX_,ObjY_,ObjZ_);
		btTransform tf = btTransform(quat,vec);

		env = new Object(m_guiHelper,m_dynamicsWorld, &m_collisionShapes,tf,parameters->file_env,envColor,btScalar(parameters->OBJ_SCALE),btScalar(0),COL_ENV,envCollidesWith);
	}

	// create object from 3D scan at a specific position and orientation using 3D quaternion (roll,pitch,yaw)
	else if(parameters->OBJECT==5){
		// add environment to world
		btVector4 envColor = btVector4(0.6,0.6,0.6,1);

		// object position
		ObjX_ = parameters->ObjX;
		ObjY_ = parameters->ObjY;
		ObjZ_ = parameters->ObjZ;

		// object orientation with 3D quaternion
		ObjYAW_ = parameters->ObjYAW;
		ObjPITCH_ = parameters->ObjPITCH;
		ObjROLL_ = parameters->ObjROLL;

		// make sure params are adjusted accordingly with the user input
		// std::cout << "ObjX: " << ObjX_ << std::endl;

		//set Quaternion and Vector 
		btQuaternion quat = btQuaternion();
		quat.setEulerZYX(ObjYAW_,ObjPITCH_,ObjROLL_);
		btVector3 vec = btVector3(ObjX_,ObjY_,ObjZ_);
		btTransform tf = btTransform(quat,vec);

		env = new Object(m_guiHelper,m_dynamicsWorld, &m_collisionShapes,tf,parameters->file_env,envColor,btScalar(parameters->OBJ_SCALE),btScalar(0),COL_ENV,envCollidesWith);
	}

	// moving object
	else if (parameters->OBJECT==6){
		// add environment to world
		btVector4 envColor = btVector4(0.6,0.6,0.6,1);

		// object position
		ObjX_ = parameters->ObjX;
		ObjY_ = parameters->ObjY;
		ObjZ_ = parameters->ObjZ;

		// object orientation with 3D quaternion
		ObjYAW_ = parameters->ObjYAW;
		ObjPITCH_ = parameters->ObjPITCH;
		ObjROLL_ = parameters->ObjROLL;

		//set Quaternion and Vector 
		btQuaternion quat = btQuaternion();
		quat.setEulerZYX(ObjYAW_,ObjPITCH_,ObjROLL_);
		btVector3 vec = btVector3(ObjX_,ObjY_,ObjZ_);
		btTransform tf = btTransform(quat,vec);

		env = new Object(m_guiHelper,m_dynamicsWorld, &m_collisionShapes,tf,parameters->file_env,envColor,btScalar(parameters->OBJ_SCALE),btScalar(0),COL_ENV,envCollidesWith);

	}


	// generate graphics
	m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);

	// set camera position to rat head
	camPos[0] = rathead_pos[0]+parameters->CPOS[0];
	camPos[1] = rathead_pos[1]+parameters->CPOS[1];
	camPos[2] = rathead_pos[2]+parameters->CPOS[2];
	camDist = parameters->CDIST;
	camPitch = parameters->CPITCH;
	camYaw = parameters->CYAW;
	resetCamera();

	// if active whisking, load whisking protraction angle trajectory
	if (parameters->ACTIVE){
		read_csv_float(parameters->dir_param + parameters->file_whisking_angle, parameters->WHISKER_VEL);
		parameters->TIME_STOP = (parameters->WHISKER_VEL[0].size()/3 - 1) * parameters->TIME_STEP;
	}

	// if exploring, load data for rat head trajectory
	if (parameters->EXPLORING){
		read_csv_float(parameters->dir_rathead_trajectory, parameters->HEAD_LOC_VEL);
	}

	if(parameters->CONNECT){
		connection.start();
	}

	// initialize time/step tracker
	m_time_elapsed = 0;
	m_time = 0;
	m_step = 0;
	
	std::cout << "\n\nStart simulation..." << std::endl;
	std::cout << "\n====================================================\n" << std::endl;
}

output* Simulation::get_results(){
	return data_dump;
}

void Simulation::resetCamera(){	
	m_guiHelper->resetCamera(camDist,camYaw,camPitch,camPos[0],camPos[1],camPos[2]);
}