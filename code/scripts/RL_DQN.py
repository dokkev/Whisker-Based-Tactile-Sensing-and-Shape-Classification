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
import seaborn as sns; sns.set_theme()


## initialze the neural networks
n_actions = 5  # dim of output layer 
input_dim = 18 # dim of input layer 
model = Sequential()
model.add(Dense(216, input_dim = input_dim , activation = 'relu'))
# model.add(Dense(128, input_dim = input_dim , activation = 'relu'))
model.add(Dense(108, activation = 'relu'))
# model.add(Dense(108, activation = 'relu'))
model.add(Dense(n_actions, activation = 'linear'))
model.compile(optimizer='adam', loss = 'mse')
print(model.summary())

## Parameters for Reinforcement Learning
n_episodes = 1000
gamma = 0.01
epsilon = 0.5
minibatch_size = 32
reward_sum_array = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000


def obtain_weights(model):
    weights = model.get_weights()
    # weights connecting input layer to hidden layer 1
    # print(weights[0])
    # bias of the hidden layer 1
    print(weights[1])
    # weights connecting hidden layer 1 to the output layer
    print(weights[2])
    # bias of the output layer
    # print(weights[3])

    return weights[0],weights[2]

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
    model.save("DQN_model_contact")

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
    if reward > 5.0:
        reward += 10.0
    return reward

def symmetic_18contact(data):
    reward = 0
    if data[0] & data[9] == 1:
        reward += 2.5
    if data[1] & data[10] == 1:
        reward += 2.5
    if data[2] & data[11] == 1:
        reward += 2.5
    if data[3] & data[12] == 1:
        reward += 2.5
    if data[4] & data[13] == 1:
        reward += 2.5
    if data[5] & data[14] == 1:
        reward += 2.5
    if data[6] & data[15] == 1:
        reward += 2.5
    if data[7] & data[16] == 1:
        reward += 2.5
    if data[8] & data[17] == 1:
        reward += 2.5

    if reward > 5.0:
        reward += 10.0
    return reward



def all_symmetic_contact(data):
    reward = 0
    if data[0] & data[27] == 1:
        reward += 2.5
    if data[1] & data[28] == 1:
        reward += 2.5
    if data[2] & data[29] == 1:
        reward += 2.5
    if data[3] & data[30] == 1:
        reward += 2.5
    if data[4] & data[31] == 1:
        reward += 2.5
    if data[5] & data[32] == 1:
        reward += 2.5
    if data[6] & data[33] == 1:
        reward += 2.5
    if data[7] & data[34] == 1:
        reward += 2.5
    if data[8] & data[35] == 1:
        reward += 2.5
    if data[9] & data[36] == 1:
        reward += 2.5
    if data[10] & data[37] == 1:
        reward += 2.5
    if data[11] & data[38] == 1:
        reward += 2.5
    if data[12] & data[39] == 1:
        reward += 2.5
    if data[13] & data[40] == 1:
        reward += 2.5
    if data[14] & data[41] == 1:
        reward += 2.5
    if data[15] & data[42] == 1:
        reward += 2.5
    if data[16] & data[43] == 1:
        reward += 2.5
    if data[17] & data[44] == 1:
        reward += 2.5
    if data[18] & data[45] == 1:
        reward += 2.5
    if data[19] & data[46] == 1:
        reward += 2.5
    if data[20] & data[47] == 1:
        reward += 2.5
    if data[21] & data[48] == 1:
        reward += 2.5
    if data[22] & data[49] == 1:
        reward += 2.5
    if data[23] & data[50] == 1:
        reward += 2.5
    if data[24] & data[51] == 1:
        reward += 2.5
    if data[25] & data[52] == 1:
        reward += 2.5
    if data[26] & data[53] == 1:
        reward += 2.5
    if reward > 5.0:
        reward += 30.0
    return reward
if __name__ == '__main__':

    WINSIZE = (720, 960)

    ## zmq setting
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    unpacker = msgpack.Unpacker()
    packer = msgpack.Packer()
    
    ## initial condition of the rat
    turn_size = 0.005
    step_size = 0.001
    state = np.array([0.0,5.0,0.0,0.0,0.0,0.0])


    ## graph in pygame
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    screen.fill((255,255,255))
    graph = Graph(screen)
    graph.add_subplots(3,1)
    graph.ylabel(0,'Reward')
    graph.ylabel(1,'Learning Rate')
    graph.ylabel(2,'Greedy Factor')
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
            state = np.array([0.,0.,0.,0.,0.,0.])


            reward_sum = 0
            reset_state = False


            while not done:
                # get info from c++
                unpacker.feed(socket.recv())
                # empty array to store dynamic data
                Y = []

                # Let's unpack the data we recievd
                for values in unpacker:
                        Y.append(np.array(values))
                
                my = np.array(Y[4]).flatten()
                mz = np.array(Y[5]).flatten()

                binary_contact_indicator = process_contact(Y,local_counter)
                
                contact_reward = np.sum(np.array(binary_contact_indicator))
                sym_reward = symmetic_18contact(binary_contact_indicator)

                real_time_reward = 0.1 * contact_reward + sym_reward

                graph.plot(0,t,real_time_reward,color=RED)
                graph.plot(1,t,gamma,color=BLUE)
                graph.plot(2,t,epsilon,color=GREEN)
                graph.update()
                    
                ## DQN ##
                # s = (my**2 + mz**2)**0.5 # whisker state before action
                s = binary_contact_indicator
    
                # Feedforward pass for current state to get predicted q-values for all actions 
                qvals_s = model.predict(s.reshape(1,input_dim))
          

                if 55 < local_counter < 65:
                    gamma = 0.99
                    do_RL = True
                else:
                    gamma = 0.01
                    do_RL = False
          
                if do_RL:
                    # Choose action to be epsilon-greedy
                    if np.random.random() < epsilon:
                        a = np.random.randint(0,high=5)
                        print("random action:" , a)
                    else:                             
                        a = np.argmax(qvals_s)
                        print("best action:" , a)
    
                # Take step, store results
                
                if do_RL:
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
                    # don't move
                    elif a == 4:
                        print("stationary")
                  

                    # sprime = (my**2 + mz**2)**0.5 # whisker state after action
                    sprime = binary_contact_indicator                    
               
                    # evaluate the reward real-time
                    reward = real_time_reward

                    if reward == 0:
                        reward = -1
                    
            
                    # add to memory, respecting memory buffer limit 
                    if len(replay_memory) > mem_max_size:
                        replay_memory.pop(0)
                    replay_memory.append({"s":s,"a":a,"r":reward,"sprime":sprime,"done":done})
                    # Update state
                    s=sprime
                    # Train the nnet that approximates q(s,a), using the replay memory
                    model=replay(replay_memory, minibatch_size = minibatch_size)

                    # print weight of each neurons
                    # weights = model.get_weights()
                    # print("weights: ",weights)

                    # Decrease epsilon until we hit a target threshold 
                    if epsilon > 0.01:      
                        epsilon -= 0.00001
                    
                    if gamma > 0.01:
                        gamma -= 0.0001
                    print("reward: ", reward)

               

                    reward_sum += reward

                if local_counter < 124:
                    symmetry_array.append(sym_reward)
                    contact_array.append(contact_reward)

              
                # sum of symmetry reward for one cycle of whisk
                sum_sym_reward = np.sum(np.array(symmetry_array))
                # sum of contact reward for one cycle of whisk
                sum_contact_reward = np.sum(np.array(contact_array))
                
                # evaluate the reward every cycle
                if local_counter == 124:
                    # empty array for the next cycle
                    symmetry_array = []
                    contact_array = []
                    longterm_reward = sum_sym_reward + (0.1 * sum_contact_reward)
                    print("long term reward: ", longterm_reward)
                    if longterm_reward < 10:
                        done = True
                        reward = -100
                        print("====================reset===================")
                 
                    
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
                if move_counter < 124:
                    move_counter += 1
                elif move_counter == 20:
                    move_counter = 0
                  


