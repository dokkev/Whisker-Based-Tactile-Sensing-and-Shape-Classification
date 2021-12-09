# General
import gym
import pandas as pd
import numpy as np
# Neural network 
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import Adam
# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns 

env = gym.make("CartPole-v1")

## Building the nnet that approximates q 
n_actions = env.action_space.n  # dim of output layer 
input_dim = env.observation_space.shape[0]  # dim of input layer 
print(input_dim)
model = Sequential()
model.add(Dense(128, input_dim = input_dim , activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(n_actions, activation = 'linear'))
model.compile(optimizer='adam', loss = 'mse')
print(model.summary())


weights = model.get_weights()
# print(weights[0].shape)
# print(weights[1].shape)
# print(weights[2].shape)
# print(weights[3].shape)
# weights connecting input layer to hidden layer 1
print(weights[0])
# bias of the hidden layer 1
# print(weights[1])
# weights connecting hidden layer 1 to the output layer
print(weights[2])
# bias of the output layer
# print(weights[3])

n_episodes = 1000
gamma = 0.99
epsilon = 0.99
minibatch_size = 32
r_sums = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000

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


for n in range(n_episodes): 
    s = env.reset()
  
    done=False
    r_sum = 0
    while not done: 
        # Uncomment this to see the agent learning
        env.render()
        
        # Feedforward pass for current state to get predicted q-values for all actions 
        qvals_s = model.predict(s.reshape(1,4))

        # Choose action to be epsilon-greedy
        if np.random.random() < epsilon:  
            a = env.action_space.sample()
        else:                             
            a = np.argmax(qvals_s); 
        # Take step, store results
       
        sprime, r, done, info = env.step(a)
        r_sum += r 
        # add to memory, respecting memory buffer limit 
        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)
        replay_memory.append({"s":s,"a":a,"r":r,"sprime":sprime,"done":done})
        # Update state
        s=sprime
        # Train the nnet that approximates q(s,a), using the replay memory
        model=replay(replay_memory, minibatch_size = minibatch_size)
        weights = model.get_weights()
       
        # Decrease epsilon until we hit a target threshold 
        if epsilon > 0.01:      
            epsilon -= 0.05

    print("Total reward:", r_sum)
    r_sums.append(r_sum)

