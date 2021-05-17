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
    
    objects = ['scan_01.obj','scan_02.obj','scan_03.obj','scan_04.obj','scan_05.obj']
    # print('Simulating: ' + str(whisker))
    
    filename = 'O' + str(objID) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d') #curr_time.replace(":","-")
    dirout = "scan1-5/"+filename
    
    # print(dirout)
    objFile = objects[objID]
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cmdstr = "~/Final_Project/whiskitphysics/code/build/whiskit \
    --PRINT 2 \
    --CDIST 50 \
    --CPITCH -0 \
    --CYAW 180 \
    --BLOW 1 \
    --WHISKER_NAMES R \
    --DEBUG 0 \
    --OBJECT 5 \  "
    "--file_env ~Final_Project/whiskitphysics/code/data/environment/"+objFile+ \
    "--dir_out ~Final_Project/whiskitphysics/code/output/"+dirout+ \
    "--ACTIVE 1 \
    --TIME_STOP 1.0 \
    --SAVE_VIDEO 0 \
    --SAVE 1  \ "  
    "--ObjX " + str(ObjX)+\
    "--ObjY " + str(ObjY)+\
    "--ObjZ " + str(ObjZ)+\
    "--ObjYAW " + str(ObjYAW)+\
    "--ObjPITCH " + str(ObjPITCH)+\
    "--ObjROLL " + str(ObjROLL) 

    start = time.time()
    s = subprocess.getoutput([cmdstr])
    time_elapsed = time.time()-start
    # print("Elapsed time: " + str(time_elapsed))

    outputlist = s.split("\n")
    collision = bool(np.sum(get_output("C: ",outputlist)))
    file = open(dirout+"/parameters.txt",'w+')
    file.write("Whiskers: "+whisker)
    file.write("\nX : "+str(ObjX))
    file.write("\nY : "+str(ObjY))
    file.write("\nZ : "+str(ObjZ))
    file.write("\nY : "+str(ObjYAW))
    file.write("\nP : "+str(ObjPITCH))
    file.write("\nR : "+str(ObjROLL))
    file.write("\nobject ID: "+str(objID))
    file.write("\nfilename: "+str(objFile))
    file.write("\ncollision: "+str(collision))
    file.write("\nsimulation time: "+str(time_elapsed))
    file.write("\ntimestamp: "+str(datetime.datetime.now()))
    file.close()


def simulate_obj(sim_input):

    objID = sim_input[0]
    trialID = sim_input[1]


    simID = 0
    for i in range(1): #the 4 allows for the program to use all 90 degree rotation of the object around the y axis

       	#runs the simulation for three different z positions
    #   simulate(whisker, ObjX, ObjY, ObjZ, ObjYAW, ObjPITCH, ObjROLL,objID,trialID,simID):   
        simulate("R", 60, 10, 10, 1.57, 0, 0,   0,trialID,simID)
        simID+=1
        simulate("R", 60, 10, 10, 1.57, 0, 0,   1,trialID,simID)
        simID+=1
        simulate("R", 60, 10, 10, 0, 0, 1.57,   2,trialID,simID)
        simID+=1
        simulate("R", 60, 10, 10, 1.57, 0, 0,   3,trialID,simID)
        simID+=1
        simulate("R", 60, 10, 10, 1.57, 0, 0.1, 4,trialID,simID)
        simID+=1


def test(x):
    global counter
    with counter.get_lock():
        counter.value += 1
    print(counter.value)
    return 0


if __name__ == "__main__":

    # global counter
    # trialbase = int(sys.argv[1])
    trialbase = 1
    counter = Value('i',trialbase)
    numConfig = 100
    trials = []
    for n in range(numConfig):
        trials.append([2,trialbase+n])

    for n in range(numConfig):
        trials.append([3,trialbase+n])

    for n in range(numConfig):
       	trials.append([4,trialbase+n])

    
    pool = Pool(processes=10,initializer = init, initargs = (counter, ))
    try:
        i = pool.map_async(simulate_obj, trials, chunksize = 1)
        i.wait()
        # print(i.get())
    finally:
        pool.close()
        pool.join()
    
    
