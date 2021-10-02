import numpy as np
from numpy.random import uniform as randu # makes it easier to read if you have one name for function
from numpy.linalg import norm # you can import just the function, no need to import the whole set (needs more memory)
import subprocess
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
def simulate(whisker, ObjX, ObjY, ObjZ, ObjYAW, ObjPITCH, ObjROLL,objID,trialID,simID):
    global counter
    #with counter.get_lock():
    #    counter.value += 1
    
 
    objects = ['concave20.obj','concave22.obj','concave24.obj','concave26.obj','concave28.obj',
               'concave30.obj','concave32.obj','concave34.obj','concave36.obj','concave38.obj',
               'concave40.obj',
               'convex20.obj','convex22.obj','convex24.obj','convex26.obj','convex28.obj',
               'convex30.obj','convex32.obj','convex34.obj','convex36.obj','convex38.obj',
               'convex40.obj']
               
        
    
    
    # print('Simulating: ' + str(whisker))
    objFile = objects[objID]


    # here's where you set directory for the output
    filename =  str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d') #curr_time.replace(":","-")
    dirout = "../output/"+str(object_type)+"/"+filename
    # dirout = "data_parameters"+filename
    
    # print(dirout)
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # ~/Final_Project/whisker_project/code/build/whiskit
    # change this path accordingly 
    str1 = "../build/whiskit_gui \
    --PRINT 2 \
    --CDIST 50 \
    --SIM_TIME 0.1 \
    --SAVE_KINEMATICS 0 \
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

    cmdstr = str1+str2+str3+strx+stry+strz+stryaw+strpitch+strroll

    start = time.time()
    print("===========NEXT SIMULATION==============")
    print("starting whiskit:" + filename)
    print("Object Type: " + str(object_type))
    print("Now Whisking: " + str(objFile))
    print("X = " + str(ObjX) + " Y = " + str(ObjY) + " Z = " +str(ObjZ))
    print("YAW = " + str(ObjYAW) + " PITCH = " + str(ObjPITCH) + " ROLL = " +str(ObjROLL))

    s = subprocess.getoutput([cmdstr])
    print("ended whiskit:" + filename)
    time_elapsed = time.time()-start
    print("Elapsed time: " + str(time_elapsed))
    # print("Simulation Object: " + str(objFile))
    outputlist = s.split("\n")
    collision = bool(np.sum(get_output("C: ",outputlist)))
    file = open(dirout+"/parameters.txt",'w+')
    file.write("Whiskers: "+whisker)
    file.write("\nX : "+str(ObjX))
    file.write("\nY : "+str(ObjY))
    file.write("\nZ : "+str(ObjZ))
    file.write("\nYAW : "+str(ObjYAW))
    file.write("\nPITCH : "+str(ObjPITCH))
    file.write("\nROLL : "+str(ObjROLL))
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
    obj_tag_max = 0 # max object tag you want to end with 
    concave_max = 10 # object tag you want to end with 
    convex_max = 21 # object tag you want to end with

    for i in range(1):

        ## random translation and rotation will be applied to cylinders ##

        # while loop  is used to make random applied differnt every siumlation
        while obj_tag <= obj_tag_max:

            if obj_tag <= concave_max: 
                # Concave
                ObjXi = 30
                ObjYi = 30
                ObjZi = -10
                ObjYAWi = 0
                ObjPITCHi = 0
                ObjROLLi = 0.3
                object_type = 'concave'
           
            elif concave_max < obj_tag <= convex_max:
                # Convex
                ObjXi = 40
                ObjYi = 10
                ObjZi = -10
                ObjYAWi = 0
                ObjPITCHi = 0
                ObjROLLi = -3.14
                object_type = 'convex'
        


            # tiny translation max delta =~ 1 cm is applied
            ObjX = round(ObjXi + random.uniform(-1.0, 1.0),4)
            ObjY = round(ObjYi + random.uniform(-1.0, 1.0),4)
            ObjZ = round(ObjZi) #+ random.uniform(-0.01, 0.01),4)

            # tiny rotation mat theta =~ 0.15 rad is applied
            ObjYAW = round(ObjYAWi) #+ random.uniform (-0.35,0.35),4)
            ObjPITCH =round(ObjPITCHi) #+ random.uniform (-0.35,0.35),4)
            ObjROLL = round(ObjROLLi + random.uniform (-0.15,0.15),4)

            simulate("R", ObjX, ObjY, ObjZ, ObjYAW, ObjPITCH, ObjROLL, obj_tag, trialID, simID)

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
    trialbase = 0
    counter = Value('i',trialbase)
    numConfig = 1 # how many times you want to simulate
    trials = []
    for n in range(numConfig):
        trials.append([2,trialbase+n])

    # for n in range(numConfig):
    #     trials.append([3,trialbase+n])

    # for n in range(numConfig):
    #    	trials.append([4,trialbase+n])

    
    pool = Pool(processes=28,initializer = init, initargs = (counter, ))
    try:
        i = pool.map_async(simulate_obj, trials, chunksize = 1)
        i.wait()
        # print(i.get())
    finally:
        pool.close()
        pool.join()
    
    