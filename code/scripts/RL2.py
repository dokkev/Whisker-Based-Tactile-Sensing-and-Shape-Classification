#!/usr/bin/env python3

from threading import local
import numpy as np
import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys
from graph import *
import pandas as pd

"""

"""
# RL Settings
RL_status = 1
#define training parameters
epsilon = 0.1 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn
# Freedom of Pitch Movement
environment_rows = 9
# Freedom of Yaw Movement
environment_columns = 15
# Actions rat can take
actions = ['up', 'right', 'down', 'left']
# Initialize the Q-table
q_values = np.zeros((environment_rows, environment_columns, 4))
# Initialize the reward matrix
rewards = np.full((environment_rows, environment_columns), -1)
# right/left array
rat_hstate = np.linspace(90,-90,num=environment_columns,dtype=int)
# up/down array
rat_vstate = np.linspace(0,-45,num=environment_rows,dtype=int)

print(rat_hstate)
print(rat_vstate)


def almost_equal(a,b):
    return np.abs(a-b) < 0.001

def is_8symmetric(array):
    rewards = 0
    if array[0] != 0 & array[4] != 0:
        rewards += 10
    if array[1] != 0 & array[5] != 0:
        rewards += 10
    if array[2] != 0 & array[6] != 0:
        rewards += 10
    if array[3] != 0 & array[7] != 0:
        rewards += 10
    return rewards
    
def save_q_values(filename,data):
    # Write the array to disk
    with open(filename, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

def load_q_values(filename):
    new_data = np.loadtxt(filename)
    new_data = new_data.reshape((9,15,4))

    return new_data


#define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
  #if the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
  if rewards[current_row_index, current_column_index] == -1:
    return False
  else:
    return True

#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
  #if a randomly chosen value between 0 and 1 is less than epsilon, 
  #then choose the most promising value from the Q-table for this state.
    
    if np.random.random() < epsilon:
        # print("qmax",np.argmax(q_values[current_row_index, current_column_index]))
        epsilon +=0.01
        return np.argmax(q_values[current_row_index, current_column_index])
    else: #choose a random action
        return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):

    new_row_index = current_row_index
    new_column_index = current_column_index
    if action_index == 0 and current_row_index > 0:
        new_row_index -= 1
    elif action_index == 1 and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif action_index == 2 and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif action_index == 3 and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


#define a function that will get the next location based on the chosen action
def get_next_location2(current_row_index, current_column_index, action_index):

    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

#define a function that will choose a random, non-terminal starting location
def get_starting_location():
  #get a random row and column index
  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
  #continue choosing random row and column indexes until a non-terminal state is identified
  #(i.e., until the chosen state is a 'white square').
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index

def get_shortest_path(start_row_index, start_column_index):
  #return immediately if this is an invalid starting location
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else: #if this is a 'legal' starting location
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        #continue moving along the path until we reach the goal (i.e., the item packaging location)
        while not is_terminal_state(current_row_index, current_column_index):
            #get the best action to take
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            #move to the next location on the path, and add the new location to the list
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
    return shortest_path




def get_Rx(theta):
    R = np.array([[1.,0.,0.],[0., np.cos(theta), -np.sin(theta)],[0.,np.sin(theta), np.cos(theta)]])
    return R

def get_Ry(theta):
    R = np.array([[np.cos(theta),0.,np.sin(theta)],[0., 1., 0.],[-np.sin(theta),0., np.cos(theta)]])
    return R

def get_Rz(theta):
    theta = theta
    R = np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta), np.cos(theta), 0.],[0.,0., 1.]])
    return R

def update_roll(state,turn_size,next_step,orientation):
    state[5] += turn_size

    if np.abs(state[5])>=2*np.pi:
        state[5] = 0.
    

    # get roll
    Rx = get_Rx(state[5])

    next_step = np.dot(Rx,orientation)
    # next_step = np.dot(Rx,next_step)
    return next_step

def update_yaw(state,turn_size,next_step,orientation):
    
    state[3] += turn_size

    if np.abs(state[3])>=2*np.pi:
        state[3] = 0.

    # get yaw
    Rz = get_Rz(state[3])

    next_step = np.dot(Rz,orientation)

    return next_step

float_formatter = "{:.5f}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})

class Communicator:
    def __init__(self):

        # Publish whisker data ROS topic
        self.protraction_status = 0
        self.contact_sum = []
        self.do_RL = 0
    
       
    def publish_whisker_data(self,data,counter):
        fx = np.array(data[0]).flatten()
        fy = np.array(data[1]).flatten()
        fz = np.array(data[2]).flatten()
        mx = np.array(data[3]).flatten()
        my = np.array(data[4]).flatten()
        mz = np.array(data[5]).flatten()
        c = np.array(data[6])
        c_flat = c.flatten()
        x = np.array(data[7]).flatten()
        y = np.array(data[8]).flatten()
        z = np.array(data[9]).flatten()

        self.whisker_force = np.array([fx,fy,fz])
        self.whisker_moment = np.array([mx,my,mz])
        self.whisker_position = np.array([x,y,z])
        self.counter = counter

        # sum contact along segments
        contact_indicator = np.sum(c,axis=1).astype(int)
        self.multi_contact_indicator = contact_indicator

        for i in range(len(contact_indicator)):
            if contact_indicator[i] >= 1:
                contact_indicator[i] = int(1)
            else:
                contact_indicator[i] = int(0)

        self.binary_contact_indicator = contact_indicator
        self.sym_reward = is_8symmetric(contact_indicator)


        mz_contact_detector = np.vstack((mz,contact_indicator))
        mz_contact_detector = np.transpose(mz_contact_detector)
        # print(mz_contact_detector)
        
        # summation of binary contact for one cycle of whisking
        if self.counter < int(125):
            self.contact_sum.append(list(self.binary_contact_indicator))
            # print(len(self.contact_sum))

        if self.counter < int(63):
            self.protraction_status = 1
        if self.counter > int(62):
            self.protraction_status = 0
        if self.counter == int(124):
            contact_sum = np.array(self.contact_sum)
            contact_sum = np.sum(contact_sum,axis=0)
            self.sum_of_binary_contact = contact_sum
            
            self.contact_sum = []

        # publish contact status
        contact_status = np.sum(c_flat)
        if contact_status == 0:
            contact_status = 0
        else:
            contact_status = 1
        self.contact_status = contact_status


    def process_contact(self,data):
        c = np.array(data[6])
        x = np.array(data[7])
        y = np.array(data[8])
        z = np.array(data[9])

        contact_x, contact_y, contact_z = [],[],[]
        for i in range(len(c)):
            for j in range(len(c[0])):
    
                # if the collision data is 1, then we will extract the position of the contact
                if int(c[i,j]) == int(1):
                    contact_x.append(float(x[i,j]))
                    contact_y.append(float(y[i,j]))
                    contact_z.append(float(z[i,j]))

        self.contact_position = np.array([contact_x,contact_y,contact_z])


    def reset_counter(self):
        self.counter = 0

    def main(self):
        WINSIZE = (720, 960)
        # initialize the node
 
        context = zmq.Context()
    
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")

        unpacker = msgpack.Unpacker()
        packer = msgpack.Packer()
        
        # initial condition
        state = np.array([0.,5.,0.,0.,0.,0.])
        row_index, column_index = 0, 7

        pygame.init()
        screen = pygame.display.set_mode(WINSIZE)
        screen.fill((255,255,255))
        graph = Graph(screen)
        graph.add_subplots(8,1)
        graph.ylabel(0,'RC0')
        graph.ylabel(1,'RC1')
        graph.ylabel(2,'RC2')
        graph.ylabel(3,'RC3')
        graph.ylabel(4,'LC0')
        graph.ylabel(5,'LC1')
        graph.ylabel(6,'LC2')
        graph.ylabel(7,'LC3')
        graph.xlabel(7,'time [s]')

        pygame.key.set_repeat(1, 10)
        turn_size = 0.02
        step_size = 0.3
        orientation = np.array([0.,step_size,0.])
        pitchaxis = np.array([0.,0.,0.])
        global next_step
        next_step = np.array([0.,step_size,0.])
        local_counter = 0
        move_counter = 0
        sym_reward_list = []
        print("And let's go!!!")
        t = 0
        while True:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    
                    # go backward
                    if event.key == pygame.K_s:
                        state[0:3] -= next_step

                    # go forward
                    if event.key == pygame.K_w:
                        state[0:3] += next_step

                    # look up
                    if event.key == pygame.K_UP:
                        next_step = update_roll(state,turn_size,next_step,orientation)
                    
                    # look down
                    if event.key == pygame.K_DOWN:
                        next_step = update_roll(state,-turn_size,next_step,orientation)

                    # turn right
                    if event.key == pygame.K_d:
                        next_step = update_yaw(state,-turn_size,next_step,orientation)
            
                    # turn left
                    if event.key == pygame.K_a:
                        next_step = update_yaw(state,turn_size,next_step,orientation)
    

            # get info from c++
            unpacker.feed(socket.recv())
            # empty array to store dynamic data
            Y = []

            # Let's unpack the data we recievd
            for values in unpacker:
                    Y.append(np.array(values))
        
            # since we are getting real-time data, we only have one row while columna represent whiskers
            # publish whisker data
            self.publish_whisker_data(Y,local_counter)
            self.process_contact(Y)


            fx = np.array(Y[0]).flatten()
            fy = np.array(Y[1]).flatten()
            fz = np.array(Y[2]).flatten()

            mx = np.array(Y[3]).flatten()
            my = np.array(Y[4]).flatten()
            mz = np.array(Y[5]).flatten()

            color_mat = [RED,BLUE,GREEN,YELLOW,RED,BLUE,GREEN,YELLOW]

            graph.plot(0,t,mz[0],color=RED)
            graph.plot(1,t,mz[1],color=BLUE)
            graph.plot(2,t,mz[2],color=GREEN)      
            graph.plot(3,t,mz[3],color=BLACK)
            graph.plot(4,t,mz[4],color=RED)
            graph.plot(5,t,mz[5],color=BLUE)
            graph.plot(6,t,mz[6],color=GREEN)
            graph.plot(7,t,mz[7],color=BLACK)
            graph.update()


            ########
            ## RL ##
            ########
            #continue taking actions (i.e., moving) until we reach a terminal state
            #(i.e., until we reach the item packaging area or crash into an item storage location)
  
           
            # if RL_status == 1 and local_counter < (60*2/3) or local_counter > 75:
            if RL_status == 1:
                #choose which action to take (i.e., where to move next)

                if local_counter == 0:

                    action_index = get_next_action(row_index, column_index, epsilon)
            
                    #perform the chosen action, and transition to the next state (i.e., move to the next location)
                    old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
                    row_index, column_index = get_next_location(row_index, column_index, action_index)

       
                if local_counter < 60:
                    if action_index == 0:
                        state[4] -= np.deg2rad(0.0833)
                    elif action_index == 1:
                        state[3] -= np.deg2rad(0.2)
                    elif action_index == 2:
                        state[4] += np.deg2rad(0.0833)
                    elif action_index == 3:
                        state[3] += np.deg2rad(0.2)
                    else:
                        break

                
                sym_reward_list.append(self.sym_reward)
                #receive the reward for moving to the new state, and calculate the temporal difference
                ## reward for one cycle
                # reward = np.array(self.sum_of_binary_contact).flatten()
                ## real-time rewards
                if local_counter == 124:
                    reward = np.array(self.sum_of_binary_contact).flatten()
                    sym_reward = np.array(sym_reward_list).flatten()
                    reward = np.sum(sym_reward) + (0.1* np.sum(reward))
                    print("current reward: ",reward)
                    print("state_map & action",row_index,column_index,action_index)
                    print("state: ", state)
                    # store the reward
                    rewards[row_index, column_index] = reward
                    old_q_value = q_values[old_row_index, old_column_index, action_index]
                    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
                    #update the Q-value for the previous state and action pair
                    new_q_value = old_q_value + (learning_rate * temporal_difference)
                    q_values[old_row_index, old_column_index, action_index] = new_q_value
                    print("new Q-value:", new_q_value, "action index: ",action_index)
                    save_q_values("Q-value.csv",q_values)
                    

              
            

                # print('Training complete!')
            
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

            

if __name__ == '__main__':
    C = Communicator()
    C.main()


