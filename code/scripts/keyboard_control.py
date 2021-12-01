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

"""

"""


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
        row_index, column_index = 0, 74

        pygame.init()
        screen = pygame.display.set_mode(WINSIZE)
        screen.fill((255,255,255))
        graph = Graph(screen)
        graph.add_subplots(4,1)
        graph.ylabel(0,'Mz1')
        graph.ylabel(1,'Mz2')
        graph.ylabel(2,'Mz3')
        graph.ylabel(3,'Mz4')
        graph.xlabel(3,'time [s]')

        pygame.key.set_repeat(1, 10)
        turn_size = 0.02
        step_size = 0.3
        orientation = np.array([0.,turn_size,0.])
        pitchaxis = np.array([0.,0.,0.])
        global next_step
        next_step = np.array([0.,step_size,0.])
        local_counter = 0
        move_counter = 0
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


            graph.plot(0,t,mz[0],color=RED)
            graph.update()


            X = [state]
            buffer = BytesIO()
            for x in X:
                buffer.write(packer.pack(list(x)))

            socket.send(buffer.getvalue() )

            t += 0.01


            
        



if __name__ == '__main__':
    C = Communicator()
    C.main()
