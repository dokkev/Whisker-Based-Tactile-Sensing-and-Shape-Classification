import numpy as np

import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys
import math


from graph import *

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

def update_state(state):
    if np.abs(state[5])>=2*np.pi:
        state[5] = 0.
    if np.abs(state[4])>=2*np.pi:
        state[4] = 0.               
    
    # get roll
    Rz = get_Rz(state[5])
    # get yaw
    Rx = get_Rx(state[4])

    next_step = np.dot(Rz,orientation)
    next_step = np.dot(Rx,next_step)
    next_step[0] = -next_step[0]
    



WINSIZE = (720, 960)

if __name__=="__main__":


    pathapp = "../build/whiskit_gui"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    # socket.connect("tcp://localhost:5555")

    unpacker = msgpack.Unpacker()
    packer = msgpack.Packer()
    
    # initial condition
    state = np.array([0.,0.,0.,0.,0.,0.])

    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    screen.fill((255,255,255))
    graph = Graph(screen)
    graph.add_subplots(6,1)
    graph.ylabel(0,'Fx')
    graph.ylabel(1,'Fy')
    graph.ylabel(2,'Fz')
    graph.ylabel(3,'Mx')
    graph.ylabel(4,'My')
    graph.ylabel(5,'Mz')
    graph.xlabel(5,'time [s]')
    # graph1.set_title('Forces')
    

    # graph2 = Graph()
    # graph2.set_title('Moments')
    # graph2.flush()

    pygame.key.set_repeat(1, 10)
    turn_size = 0.02
    step_size = 0.1
    orientation = np.array([0.,step_size,0.])
    pitchaxis = np.array([0.,0.,0.])
    global next_step
    next_step = np.array([0.,step_size,0.])
    
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
                    state[4] += next_step

                    # update_state(state)

                    if np.abs(state[4])>=2*np.pi:
                        state[4] = 0.
                    
                    # get roll
                    Rz = get_Rz(state[5])
                    # get yaw
                    Rx = get_Rx(state[4])

                    # next_step = np.dot(Rz,orientation)
                    next_step = np.dot(Rx,next_step)
                    # next_step[0] = -next_step[0]
                
                # look down
                if event.key == pygame.K_DOWN:
                    state[4] -= next_step
                    # update_state(state)

                    if np.abs(state[4])>=2*np.pi:
                        state[4] = 0.
                    
                    # get roll
                    Rz = get_Rz(state[5])
                    # get yaw
                    Rx = get_Rx(state[4])

                    # next_step = np.dot(Rz,orientation)
                    next_step = np.dot(Rx,next_step)
                    # next_step[0] = -next_step[0]

                # turn right
                if event.key == pygame.K_RIGHT:
                    state[5] -= turn_size

                    if np.abs(state[5])>=2*np.pi:
                        state[5] = 0.
                    
                    # get roll
                    Rz = get_Rz(state[5])
                    # get yaw
                    Rx = get_Rx(state[4])

                    next_step = np.dot(Rz,orientation)
                    # next_step = np.dot(Rx,next_step)
        

                # turn left
                if event.key == pygame.K_LEFT:
                    state[5] += turn_size

                    if np.abs(state[5])>=2*np.pi:
                        state[5] = 0.
                    
                    # get roll
                    Rz = get_Rz(state[5])
                    # get yaw
                    Rx = get_Rx(state[4])
     
                    next_step = np.dot(Rz,orientation)
                    # next_step = np.dot(Rx,next_step)
         

                # # go up
                # if event.key == pygame.K_CTRL:
                #     state[2] += step_size

                # # go down
                # if event.key == pygame.K_SPACE:
                #     state[2] -= step_size


        X = [state]
        # print("X: ", X)
        #  Wait for next request from client
        # print("Waiting for response...")

        # message = socket.recv()
        # print("Received request: ", message)

        unpacker.feed(socket.recv())
        Y = []
        for values in unpacker:
                Y.append(np.array(values))
                print(values)

        # # if t>graph1.xmax:
        # #     t = 0.
        #     # graph1.flush()
        #     # graph2.flush()

        # fx = np.array(Y[0]).flatten()
        # fy = np.array(Y[1]).flatten()
        # fz = np.array(Y[2]).flatten()

        # mx = np.array(Y[3]).flatten()
        # my = np.array(Y[4]).flatten()
        # mz = np.array(Y[5]).flatten()


        
        # if fx.shape[0]>0:

            # i=0
            # graph.plot(0,t,fx[i],color=BLUE)
            # graph.plot(1,t,fy[i],color=BLACK)
            # graph.plot(2,t,fz[i],color=RED)
            # graph.plot(3,t,mx[i],color=BLUE)
            # graph.plot(4,t,my[i],color=BLACK)
            # graph.plot(5,t,mz[i],color=RED)

            # i=23
            # graph.plot(0,t,fx[i],color=BLUE)
            # graph.plot(1,t,fy[i],color=BLACK)
            # graph.plot(2,t,fz[i],color=RED)
            # graph.plot(3,t,mx[i],color=BLUE)
            # graph.plot(4,t,my[i],color=BLACK)
            # graph.plot(5,t,mz[i],color=RED)

            # graph.update()

        buffer = BytesIO()
        for x in X:
            buffer.write(packer.pack(list(x)))

        socket.send(buffer.getvalue() )

        t += 0.01

        