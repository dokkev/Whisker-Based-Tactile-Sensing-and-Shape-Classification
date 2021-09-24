
#ifndef ERGODICEXPLORATION_HPP
#define ERGODICEXPLORATION_HPP

#include <vector>
#include <thread>
#include <memory>
#include <functional>

#include <iostream>
#include "Simulation_IO.h"

struct state{
    float x;
    float y;
    float z;
    float pitch;
    float yaw;
    float roll;
    float vel_whisk;
};

class State {

private: 
    
    int current_step;
    int final_step;
    float scale = 1.;

public:

    State(){}
    ~State(){}

    std::vector<std::vector<float>> new_state;


    void update(){
        
        current_state.x = new_state[0][0];
        current_state.y = new_state[0][1];
        current_state.z = new_state[0][2];
        current_state.pitch = new_state[0][3];
        current_state.yaw = new_state[0][4];
        current_state.roll = new_state[0][5];
        current_state.vel_whisk = new_state[0][6];
        
        new_state.clear();
    }   
    

    state get_state(){
        return current_state;
    }

    state current_state;
    
};


#endif