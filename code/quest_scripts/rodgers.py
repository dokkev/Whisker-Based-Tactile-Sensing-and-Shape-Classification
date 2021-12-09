#!/usr/bin/env python
 
import numpy as np
from numpy.random import uniform as randu # makes it easier to read if you have one name for function
from numpy.linalg import norm # you can import just the function, no need to import the whole set (needs more memory)
import subprocess
import multiprocessing
from multiprocessing import Pool, Value
from math import acos, asin
import random
import datetime
import time
import os
import pathlib
import sys
import csv
"""
This script takes care of parallel simulation.
"""
counter = None
E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args

def print_output(sub,outputlist):
    dummy = [float(st.replace(sub,'')) for st in outputlist if sub in st]
    print(dummy)

def get_output(sub,outputlist):
    dummy = [float(st.replace(sub,'')) for st in outputlist if sub in st]
    return np.array(dummy)

#changes magnitude of vector to 1
def normalize(vect):

    
    return vect/norm(vect,2) # operations on each element in a for loop is very slow -> vector operations

#runs simulation with stated parameters
def simulate(whisker,RatPOS, RatORI, ObjX, ObjY, ObjZ, ObjYAW, ObjPITCH, ObjROLL,objID,trialID,simID):
    global counter
    #with counter.get_lock():
    #    counter.value += 1
    
 
    objects = ['concave40.obj','concave40.obj','concave40.obj',
    'convex40.obj','convex40.obj','convex40.obj']
               
        
    
    
    # print('Simulating: ' + str(whisker))
    objFile = objects[objID]


    # here's where you set directory for the output
    filename =  str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d') #curr_time.replace(":","-")
    dirout = "../output/"+str("rodgers_train")+"/"+filename
    # dirout = "data_parameters"+filename
    
    # print(dirout)
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ~/Final_Project/whisker_project/code/build/whiskit
    # change this path accordingly 
    str1 = "../build/whiskit \
    --PRINT 2 \
    --CDIST 50 \
    --SIM_TIME 0.125 \
    --SAVE_KINEMATICS 0 \
    --WHISKER_NAMES RODGERS \
    --CPITCH 0 \
    --CYAW 180 \
    --BLOW 1  \
    --OBJECT 5 \
    --ACTIVE 1 \
    --SAVE_VIDEO 0 \
    --SAVE 1 "

    str2 = " --file_env ../data/concave/" + objFile
    str3 = " --dir_out " + str(dirout)
    strx = " --ObjX " + str(ObjX) 
    stry = " --ObjY " + str(ObjY)
    strz = " --ObjZ " + str(ObjZ)
    stryaw = " --ObjYAW " + str(ObjYAW)
    strpitch = " --ObjPITCH " + str(ObjPITCH)
    strroll = " --ObjROLL " + str(ObjROLL)
    strpos = " --POSITION " + str(RatPOS)
    strori = " --ORIENTATION " + str(RatORI) 

    cmdstr = str1+str2+str3+strx+stry+strz+stryaw+strpitch+strroll+strpos+strori

    start = time.time()
    print1 = ("\n===========NEXT SIMULATION==============")
    print2 = ("\nstarting whiskit:" + filename)
    print3 = ("\nObject Type: " + str(object_type))
    print4 = ("\nNow Whisking: " + str(objFile))
    print5 = ("\nObjX = " + str(ObjX) + " ObjY = " + str(ObjY) + " ObjZ = " +str(ObjZ))
    print6 = ("\nObjYAW = " + str(ObjYAW) + " ObjPITCH = " + str(ObjPITCH) + " ObjROLL = " +str(ObjROLL))
    print7 = ("\nRatPos = " + str(RatPOS))
    print8 = ("\nRatOri = " + str(RatORI))
    print(print1+print2+print3+print4+print5+print6+print7+print8)
    s = subprocess.getoutput([cmdstr])
    print("ended whiskit:" + filename)
    time_elapsed = time.time()-start
    print("Elapsed time: " + str(time_elapsed))
    # print("Simulation Object: " + str(objFile))
    outputlist = s.split("\n")
    collision = bool(np.sum(get_output("C: ",outputlist)))
    file = open(dirout+"/parameters.txt",'w+')
    file.write("Whiskers: "+whisker)
    file.write("\nObject Type: "+str(object_type))
    file.write("\nX : "+str(ObjX))
    file.write("\nY : "+str(ObjY))
    file.write("\nZ : "+str(ObjZ))
    file.write("\nObjYAW : "+str(ObjYAW))
    file.write("\nObjPITCH : "+str(ObjPITCH))
    file.write("\nObjROLL : "+str(ObjROLL))
    file.write("\nobject ID: "+str(objID))
    file.write("\ntrial ID: "+str(trialID))
    file.write("\nfilename: "+str(objFile))
    file.write("\ncollision: "+str(collision))
    file.write("\nsimulation time: "+str(time_elapsed))
    file.close()


def simulate_obj(sim_input):

    objID = sim_input[0]
    trialID = sim_input[1]

    simID = 0

    global x,y,z,yaw,pitch,roll,obj_num
    global object_type

    obj_tag = 0 # object tag you want to start with
    obj_tag_max = 5 # max object tag you want to end with 
    concave_max = 2 # object tag you want to end with 
    convex_max = 5 # object tag you want to end with

    for i in range(1):

        # while loop  is used to make random applied differnt every siumlation
        while obj_tag <= obj_tag_max:

            RatXi = 0
            RatYi = 0
            RatZi = 0
            RatYAWi = 0
            RatPITCHi = 0
            RatROLLi = 0

            if obj_tag <= concave_max: 
                # Concave
                if obj_tag == 0:
                    ObjXi = 25
                    ObjYi = 25
                if obj_tag == 1:
                    ObjXi = 26
                    ObjYi = 26
                if obj_tag == 2:
                    ObjXi = 27
                    ObjYi = 27

                ObjZi = -10
                ObjYAWi = 0.2
                ObjPITCHi = 0
                ObjROLLi = 0
                object_type = 'concave'
           
            elif concave_max < obj_tag <= convex_max:
                # Convex
                if obj_tag == 0:
                    ObjXi = 25
                    ObjYi = 25
                if obj_tag == 1:
                    ObjXi = 26
                    ObjYi = 26
                if obj_tag == 2:
                    ObjXi = 27
                    ObjYi = 27

                ObjYAWi = 3.34
                ObjPITCHi = 0
                ObjROLLi = 0
                object_type = 'convex'
        
            # Rat Position & Orientation
            RatX = round(RatXi + random.uniform(-1.0, 1.0),4)
            RatY = round(RatYi + random.uniform(-1.0, 1.0),4)
            RatZ = RatZi
            RatYAW = round(RatYAWi + random.uniform(-0.1, 0.1),4)
            RatPITCH = RatPITCHi
            RatROLL = RatROLLi
            POS = str(RatX) + " " + str(RatY) + " " + str(RatZ)
            ORI = str(RatYAW) + " " + str(RatPITCH) + " " + str(RatROLL)

            # translation of the object
            ObjX = round(ObjXi,4)
            ObjY = round(ObjYi,4)
            ObjZ = round(ObjZi,4)

            # rotation of the object
            ObjYAW = round(ObjYAWi,4) #+ random.uniform (-0.35,0.35),4)
            ObjPITCH =round(ObjPITCHi,4) #+ random.uniform (-0.35,0.35),4)
            ObjROLL = round(ObjROLLi,4)

            simulate("RODGERS",POS, ORI, ObjX, ObjY, ObjZ, ObjYAW, ObjPITCH, ObjROLL, obj_tag, trialID, simID)

            # increase simulation ID
            simID+=1

            # make it simulate next 
            obj_tag+=1



def test(x):
    global counter
    with counter.get_lock():
        counter.value += 1
    print(counter.value)
    return 0

    
if __name__ == "__main__":

    # global counter
    # trialbase = int(sys.argv[1])
    trialbase = 500
    counter = Value('i',trialbase)
    numConfig = 500 # how many times you want to simulate
    trials = []
    for n in range(numConfig):
        trials.append([2,trialbase+n])

    # for n in range(numConfig):
    #     trials.append([3,trialbase+n])

    # for n in range(numConfig):
    #    	trials.append([4,trialbase+n])

    
    pool = Pool(processes=8,initializer = init, initargs = (counter, ))

    try:
        i = pool.map_async(simulate_obj, trials, chunksize = 1)
        i.wait()
        # print(i.get())
    finally:
        pool.close()
        pool.join()