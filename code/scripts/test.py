import numpy as np
import time
import numpy as np
import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys
from graph import *

"""

"""
# RL Settings
RL_status = 1
#define training parameters
epsilon = 0.1 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn
# Freedom of Rz movement
environment_rows = 9
# Freedom of Rx movement
environment_columns = 15
# Actions rat can take
actions = ['up', 'right', 'down', 'left']
# Initialize the Q-table
q_values = np.zeros((environment_rows, environment_columns, 4))
# Initialize the reward matrix
rewards = np.full((environment_rows, environment_columns), -1)
# right/left array
rat_hstate = np.linspace(-90,90,num=15,dtype=int)
# up/down array
rat_vstate = np.linspace(45,-45,num=9,dtype=int)

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
        print("qmax",np.argmax(q_values[current_row_index, current_column_index]))
        return np.argmax(q_values[current_row_index, current_column_index])
    else: #choose a random action
        return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):

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

def main():

    # initial condition
    state = np.array([0.,0.,0.,0.,0.,0.])
    row_index, column_index = 4, 6

    local_counter = 0
    print("And let's go!!!")
    t = 0
    for i in range (1000):
        

        ########
        ## RL ##
        ########
        #continue taking actions (i.e., moving) until we reach a terminal state
        #(i.e., until we reach the item packaging area or crash into an item storage location)

        if RL_status == 1:
            #choose which action to take (i.e., where to move next)
            action_index = get_next_action(row_index, column_index, epsilon)
        
            #perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
            row_index, column_index = get_next_location(row_index, column_index, action_index)

            # update the rat state accordingly with the state map
            state[3] = np.deg2rad(rat_hstate[column_index])
            state[5] = np.deg2rad(rat_vstate[row_index])
            # print("state & action: ",state, action_index)
            
            #receive the reward for moving to the new state, and calculate the temporal difference
            # reward = np.array(self.sum_of_binary_contact).flatten()
            # reward = np.sum(reward)
            reward = np.random.uniform(0,10)
            print("current reward: ",reward)
            print("state_map & action",row_index,column_index,action_index)
            # store the reward
            rewards[row_index, column_index] = reward
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
        

        t += 0.01

        if local_counter < 125:
            local_counter += 1
        elif local_counter == 125:
            local_counter = 0


def randu():
    while True:
        print(np.random.randint(3))

if __name__ == '__main__':
    randu()