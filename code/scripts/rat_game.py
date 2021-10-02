import numpy as np
import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys
import math

from rot_mat import *
from graph import *

WINSIZE = (720, 960)

if __name__=="__main__":

    pathapp = "../build/whiskit_gui"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

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
    step_size = 0.3
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
                    next_step = update_pitch(state,turn_size,next_step,orientation)
                
                # look down
                if event.key == pygame.K_DOWN:
                    next_step = update_pitch(state,-turn_size,next_step,orientation)

                # turn right
                if event.key == pygame.K_RIGHT:
                    next_step = update_roll(state,-turn_size,next_step,orientation)
        
                # turn left
                if event.key == pygame.K_LEFT:
                    next_step = update_roll(state,turn_size,next_step,orientation)
  

        # get info from c++
        unpacker.feed(socket.recv())
        # empty array to store dynamic data
        Y = []
        for values in unpacker:
                Y.append(np.array(values))
       
        whisker_num = len(Y) - 6

        fx = np.array(Y[0]).flatten()
        fy = np.array(Y[1]).flatten()
        fz = np.array(Y[2]).flatten()
        mx = np.array(Y[3]).flatten()
        my = np.array(Y[4]).flatten()
        mz = np.array(Y[5]).flatten()
  
        # empty array to store collision data
        C = []
        i = 1
        while i <= whisker_num:
            C.append(np.array(Y[5+i]).flatten())
            i += 1
        
        
        if fx.shape[0]>0:
            
            ## Plot ALL whiskers data
            # j = 0
            # while j <= (whisker_num-1):
            #     graph.plot(0,t,fx[j],color=BLUE)
            #     graph.plot(1,t,fy[j],color=BLACK)
            #     graph.plot(2,t,fz[j],color=RED)
            #     graph.plot(3,t,mx[j],color=BLUE)
            #     graph.plot(4,t,my[j],color=BLACK)
            #     graph.plot(5,t,mz[j],color=RED)
            #     j += 1

            ## Plot ONLY one whisker data
            # i=1
            # graph.plot(0,t,fx[i],color=BLUE)
            # graph.plot(1,t,fy[i],color=BLACK)
            # graph.plot(2,t,fz[i],color=RED)
            # graph.plot(3,t,mx[i],color=BLUE)
            # graph.plot(4,t,my[i],color=BLACK)
            # graph.plot(5,t,mz[i],color=RED)

            # Plot Mean of ALL whiskers data
            graph.plot(0,t,np.nanmean(fx),color=BLUE)
            graph.plot(1,t,np.nanmean(fy),color=BLACK)
            graph.plot(2,t,np.nanmean(fz),color=RED)
            graph.plot(3,t,np.nanmean(mx),color=BLUE)
            graph.plot(4,t,np.nanmean(my),color=BLACK)
            graph.plot(5,t,np.nanmean(mz),color=RED)


            graph.update()
        # store state and send it to c++
        X = [state]
        buffer = BytesIO()
        for x in X:
            buffer.write(packer.pack(list(x)))

        socket.send(buffer.getvalue() )

        t += 0.01

        