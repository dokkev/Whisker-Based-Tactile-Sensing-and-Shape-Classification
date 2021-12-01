# General
import random
import gym
from numpy.lib.polynomial import RankWarning
import pandas as pd
import numpy as np
# Neural network 
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import Adam
# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys
from graph import *
from rotation import *
import signal

"""
"""

## Building the nnet that approximates q 
n_actions = 4  # dim of output layer 
input_dim = 6 # dim of input layer 
model = Sequential()
model.add(Dense(64, input_dim = input_dim , activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(n_actions, activation = 'linear'))
model.compile(optimizer='adam', loss = 'mse')

n_episodes = 1000
gamma = 0.99
epsilon = 1
epilson = 1
minibatch_size = 32
reward_sum_array = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000
state = np.array([0.0,5.0,0.0,0.0,0.0,0.0])




def handler(signum, frame):
    res = input("Ctrl-c was pressed, would you like to save your model? (y/n)\n")
    if res == 'y':
        save_model(model)
        exit(1)
    if res == 'n':
        exit(1)


def replay(replay_memory, minibatch_size=32):
    # choose <s,a,r,s',done> experiences randomly from the memory
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
    # create one list containing s, one list containing a, etc
    s_l =      np.array(list(map(lambda x: x['s'], minibatch)))
    a_l =      np.array(list(map(lambda x: x['a'], minibatch)))
    r_l =      np.array(list(map(lambda x: x['r'], minibatch)))
    sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
    done_l   = np.array(list(map(lambda x: x['done'], minibatch)))
    # Find q(s', a') for all possible actions a'. Store in list
    # We'll use the maximum of these values for q-update  
    qvals_sprime_l = model.predict(sprime_l)
    # Find q(s,a) for all possible actions a. Store in list
    target_f = model.predict(s_l)
    # q-update target
    # For the action we took, use the q-update value  
    # For other actions, use the current nnet predicted value
    for i,(s,a,r,qvals_sprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, done_l)): 
        if not done:  target = r + gamma * np.max(qvals_sprime)
        else:         target = r
        target_f[i][a] = target
    # Update weights of neural network with fit() 
    # Loss function is 0 for actions we didn't take
    model.fit(s_l, target_f, epochs=1, verbose=0)
    return model

def save_model(model):
    print("model saved")
    model.save("DQN_model")

def process_contact(data,counter):
    c = np.array(data[6])
    
    # sum contact along segments

    contact_indicator = np.sum(c,axis=1).astype(int)
    for i in range(len(contact_indicator)):
        if contact_indicator[i] >= 1:
            contact_indicator[i] = int(1)
        else:
            contact_indicator[i] = int(0)

    binary_contact_indicator = contact_indicator

    return binary_contact_indicator

def symmetic_contact(data):
    reward = 0
    if data[0] & data[6] == 1:
        reward += 2.5
    if data[1] & data[7] == 1:
        reward += 2.5
    if data[2] & data[8] == 1:
        reward += 2.5
    if data[3] & data[9] == 1:
        reward += 2.5
    if data[4] & data[10] == 1:
        reward += 2.5
    if data[5] & data[11] == 1:
        reward += 2.5

    return reward
if __name__ == '__main__':

    WINSIZE = (720, 960)
    # initialize the node

    context = zmq.Context()

    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    unpacker = msgpack.Unpacker()
    packer = msgpack.Packer()
    
    # initial condition
    turn_size = 0.005
    step_size = 0.001


    # graph in pygame
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    screen.fill((255,255,255))
    graph = Graph(screen)
    graph.xmin = 0.
    graph.xmax = 100
    graph.ymin = 0
    graph.ymax = 1000
    graph.add_subplots(1,1)
    graph.ylabel(0,'Reward')
    graph.xlabel(0,'time [s]')

    local_counter = 0
    move_counter = 0
    contact_sum = []

   

    print("And let's go!!!")
    t = 0
    symmetry_array = []
    contact_array = []
    
    signal.signal(signal.SIGINT, handler)

    while True:

        for i in range(n_episodes):
            
            # reset the environment
            done = False
            # change the initial state everytime it resets
       
            global next_step
            next_step = np.array([0.,step_size,0.])
            orientation = np.array([0.,turn_size,0.])
            next_step = np.array([0.,step_size,0.])
            state = np.array([0.,5.,0.,0.,0.,0.])


            reward_sum = 0


            while not done:
                # get info from c++
                unpacker.feed(socket.recv())
                # empty array to store dynamic data
                Y = []

                # Let's unpack the data we recievd
                for values in unpacker:
                        Y.append(np.array(values))

                binary_contact_indicator = process_contact(Y,local_counter)
                
                contact_reward = np.sum(np.array(binary_contact_indicator))
                sym_reward = symmetic_contact(binary_contact_indicator)

                real_time_reward = contact_reward + 0.1*sym_reward

                graph.plot(0,t,real_time_reward,color=RED)
                graph.update()
                    
                ## DQN ##
                s = state
        
                if move_counter == 0:
                    # Feedforward pass for current state to get predicted q-values for all actions 
                    qvals_s = model.predict(s.reshape(1,6))

                    # Choose action to be epsilon-greedy
                    if np.random.random() < epilson:
                        a = np.random.randint(0,high=4)
                    else:                             
                        a = np.argmax(qvals_s); 
                    # Take step, store results

                if local_counter < 60 or local_counter > 80:
                    
                    # turn right
                    if a == 0:
                        next_step = update_yaw(state,-turn_size,next_step,orientation)
                    # turn left
                    elif a == 1:
                        next_step = update_yaw(state,turn_size,next_step,orientation)
                    # look up
                    elif a == 2:
                        next_step = update_roll(state,turn_size,next_step,orientation)
                    # look down
                    elif a == 3:
                        next_step = update_roll(state,-turn_size,next_step,orientation)
                    # move forward
                    elif a == 4:
                        state[0:3] += next_step
                    # move backward
                    elif a == 5:
                        state[0:3] -= next_step

                if local_counter < 124:
                    symmetry_array.append(sym_reward)
                    contact_array.append(contact_reward)

                sprime = state
                if local_counter == 124:
                    # sum of symmetry reward for one cycle of whisk
                    sum_sym_reward = np.sum(np.array(symmetry_array))
                    # sum of contact reward for one cycle of whisk
                    sum_contact_reward = np.sum(np.array(contact_array))

                    reward = sum_sym_reward + (0.1 * sum_contact_reward)
                    
        
                    # add to memory, respecting memory buffer limit 
                    if len(replay_memory) > mem_max_size:
                        replay_memory.pop(0)
                    replay_memory.append({"s":s,"a":a,"r":reward,"sprime":sprime,"done":done})
                    # Update state
                    s=sprime
                    # Train the nnet that approximates q(s,a), using the replay memory
                    model=replay(replay_memory, minibatch_size = minibatch_size)
                    # Decrease epsilon until we hit a target threshold 
                    if epsilon > 0.01:      
                        epsilon -= 0.001
                    print("state: ", state)
                    print("reward: ", reward)

                    # empty array for the next cycle
                    symmetry_array = []
                    contact_array = []

                    reward_sum += reward

                    # if the reward is lower than 100, reset the state (next episode)
                    if reward < 100:
                        done = True
                        print("reset!")
                    
                reward_sum_array.append(reward_sum)

                X = [state]
                buffer = BytesIO()
                for x in X:
                    buffer.write(packer.pack(list(x)))

                socket.send(buffer.getvalue() )

                t += 0.01
                if local_counter < 124:
                    local_counter += 1
                elif local_counter == 124:
                    local_counter = 0
                if move_counter < 20:
                    move_counter += 1
                elif move_counter == 20:
                    move_counter = 0


        model.save('model.h5')