import numpy as np

import zmq
import msgpack
import pygame
from pygame.locals import *
from io import BytesIO
import sys

from graph import *

def get_Rx(theta):
    R = np.array([[1.,0.,0.],[0., np.cos(theta), -np.sin(theta)],[0.,np.sin(theta), np.cos(theta)]])
    return R

def get_Ry(theta):
    R = np.array([[np.cos(theta),0.,np.sin(theta)],[0., 1., 0.],[-np.sin(theta),0., np.cos(theta)]])
    return R

def get_Rz(theta):
    theta = theta
    R = np.array([[np.cos(-theta),-np.sin(-theta),0.],[np.sin(theta), np.cos(theta), 0.],[0.,0., 1.]])
    return R

def getAxis(number):
    # when nothing is moved on an axis, the VALUE IS NOT EXACTLY ZERO
    # so this is used not "if joystick value not zero"
    if joystick.get_axis(number) < -0.1 or joystick.get_axis(number) > 0.1:
      # value between 1.0 and -1.0
      print("Axis value is %s" %(joystick.get_axis(number)))
      print ("Axis ID is %s" %(number))

WINSIZE = (720, 960)

if __name__=="__main__":

    pathapp = "virtualrat_simulation/build/App_Whisker"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    unpacker = msgpack.Unpacker()
    packer = msgpack.Packer()
    
    # initial condition
    state = np.array([0.,0.,0.,0.,0.,0.,40])

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
    turn_size = 0.01
    step_size = 0.1
    angle = 0
    orientation = np.array([0.,step_size,0.])
    curr_orient = np.array([0.,1,0.])
    pitchaxis = np.array([0.,0.,0.])
    next_step = np.array([0.,step_size,0.])
    
    # how many joysticks connected to computer?
    joystick_count = pygame.joystick.get_count()
    print ("There is " + str(joystick_count) + " joystick/s")

    if joystick_count == 0:
        # if no joysticks, quit program safely
        print ("Error, I did not find any joysticks")
        pygame.quit()
        sys.exit()
    else:
        # initialise joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

    axes = joystick.get_numaxes()
    buttons = joystick.get_numbuttons()
    hats = joystick.get_numhats()

    print ("There is " + str(axes) + " axes")
    print ("There is " + str(buttons) + " button/s")
    print ("There is " + str(hats) + " hat/s")

    print("And let's go!!!")
    t = 0
    while True:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # if event.type == pygame.KEYDOWN:
            # for number in range(axes):
        # if joystick.get_button(6) == 1:
        #     state[6] = 1

        if joystick.get_button(7) == 1:
            state = np.array([0.,0.,0.,0.,0.,0.,40])
            angle = 0

        if ((joystick.get_axis(4)) > 0.1):
            # angle = (np.arccos(np.dot(orientation,curr_orient)))
            # print(angle)
            if( angle <= np.pi/2):
                next_step[2] = (joystick.get_axis(4)/100)
                state[0:3] -= next_step
                state[2] = 0.
            else:
                next_step[2] = (joystick.get_axis(4)/100)
                state[0:3] += next_step
                state[2] = 0.

        if ((joystick.get_axis(4)) < -0.1):
            # angle = (np.arccos(np.dot(orientation,curr_orient)))
            # print(angle)
            if(angle  <= np.pi/2):
                next_step[2] = (joystick.get_axis(4)/100)
                state[0:3] += next_step
                state[2] = 0.
            else:
                next_step[2] = (joystick.get_axis(4)/100)
                state[0:3] -= next_step
                state[2] = 0.
            # print("Axis value is %s" %(joystick.get_axis(number)))
            # print ("Axis ID is %s" %(number))

        # if (joystick.get_axis(4) < -0.1):
        #     # print("Axis value is %s" %(joystick.get_axis(number)))
        #     # print ("Axis ID is %s" %(number))
        #     next_step[2] = joystick.get_axis(4)/100
        #     state[0:3] += next_step
        #     state[2] = 0.
        # if (joystick.get_axis(4) > 0.1):
        #     next_step[2] = joystick.get_axis(4)/100
        #     state[0:3] -= next_step
        #     state[2] = 0.

        if (joystick.get_axis(3) < -0.1 or joystick.get_axis(3) > 0.1):
            turn_size = joystick.get_axis(3)/50
            state[4] += turn_size
            angle += turn_size
            if angle >= 2*np.pi:
                angle = 0

            if np.abs(state[4])>=2*np.pi:
                state[4] = 0.
            # print('yaw:' + str(state[4]))
            Rz = get_Rz(state[4])
            # Rx = get_Rx(state[3])
            next_step = np.dot(Rz,orientation)

        if (joystick.get_axis(1) < 0.1 or joystick.get_axis(1) > 0.1):
            turn_size = joystick.get_axis(1)/50
            state[3] -= turn_size
            
            if np.abs(state[3])>=2*np.pi:
                state[3] = 0.

            Rz = get_Rz(state[4])
            # Rx = get_Rx(state[3])1000
            next_step = np.dot(Rz,orientation)
            curr_orient = np.dot(Rz,orientation)

        if (joystick.get_axis(0) < 0.1 or joystick.get_axis(0) > 0.1):
            turn_size = joystick.get_axis(0)/50
            state[5] -= turn_size

            if np.abs(state[5])>=2*np.pi:
                state[5] = 0.

            Rz = get_Rz(state[5])
            # Rx = get_Rx(state[3])
            next_step = np.dot(Rz,orientation)
            curr_orient = np.dot(Rz,orientation)

        X = [state]
        #  Wait for next request from client
        # print("Waiting for response...")

        unpacker.feed(socket.recv())

        Y = []
        for values in unpacker:
                Y.append(np.array(values))

        fx = np.array(Y[0]).flatten()
        fy = np.array(Y[1]).flatten()
        fz = np.array(Y[2]).flatten()

        mx = np.array(Y[3]).flatten()
        my = np.array(Y[4]).flatten()
        mz = np.array(Y[5]).flatten()
        
        if fx.shape[0]>0:

            i=23
            graph.plot(0,t,fx[i],color=BLUE)
            graph.plot(1,t,fy[i],color=BLACK)
            graph.plot(2,t,fz[i],color=RED)
            graph.plot(3,t,mx[i],color=BLUE)
            graph.plot(4,t,my[i],color=BLACK)
            graph.plot(5,t,mz[i],color=RED)

            graph.update()

        buffer = BytesIO()
        for x in X:
            buffer.write(packer.pack(list(x)))

        socket.send(buffer.getvalue() )

        t += 0.01

        