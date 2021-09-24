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
def simulate(whisker, xObj, yObj, zObj, thetaObj, phiObj, zetaObj, omegaObj, xVect, yVect, zVect, speed, scale, objID,trialID,simID):
   
    global counter
    #with counter.get_lock():
    #    counter.value += 1
    
    objects = ['duck.obj','teddy.obj','plate.obj','torus_only.obj','table.obj']
    # print('Simulating: ' + str(whisker))
    
    filename = 'O' + str(objID) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d') #curr_time.replace(":","-")
    dirout = "2-obj_dataset_HFWF_3/"+filename
    
    # print(dirout)

    objFile = objects[objID]
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cmdstr = "virtual-rat/build/AppWhiskerGui \
    --SAVE 1 \
    --dir_out " + dirout+\
    " --ACTIVE 1 \
    --TIME_STOP 1 \
    --WHISK_FREQ 8 \
    --TIME_STEP 0.005  \
    --NUM_UNITS 20 \
    --NUM_STEP_INT " + str(100) + \
    " --WHISKER_NAMES " + whisker + \
    " --OBJ_POSITION "+str(xObj)+\
    " --OBJ_POSITION "+str(yObj)+\
    " --OBJ_POSITION "+str(zObj)+\
    " --OBJ_ORIENTATION "+str(thetaObj)+\
    " --OBJ_ORIENTATION "+str(phiObj)+\
    " --OBJ_ORIENTATION "+str(zetaObj)+\
    " --OBJ_ORIENTATION "+str(omegaObj)+\
    " --OBJ_SCALE "+str(scale)+\
    " --OBJ_DIR virtual-rat/data/3D_data/various/"+objFile+\
    " --SPEED " + str(speed)

    start = time.time()
    s = subprocess.getoutput([cmdstr])
    time_elapsed = time.time()-start
    # print("Elapsed time: " + str(time_elapsed))
    print(objID)
    outputlist = s.split("\n")
    collision = bool(np.sum(get_output("C: ",outputlist)))
    file = open(dirout+"/parameters.txt",'w+')
    file.write("Whiskers: "+whisker)
    file.write("\nX vector: "+str(xVect))
    file.write("\nY vector: "+str(yVect))
    file.write("\nZ vector: "+str(zVect))
    file.write("\nX object: "+str(xObj))
    file.write("\nY object: "+str(yObj))
    file.write("\nZ object: "+str(zObj))
    file.write("\nobject ID: "+str(objID))
    file.write("\nfilename: "+str(objFile))
    file.write("\ncollision: "+str(collision))
    file.write("\nsimulation time: "+str(time_elapsed))
    file.write("\ntimestamp: "+str(datetime.datetime.now()))
    file.close()


def simulate_obj(sim_input):

    objID = sim_input[0]
    trialID = sim_input[1]
    
	
    # filename = inputdata.object
    scale = randu(0.2,2)
    speed = randu(1.,3.)
    #chooses a random orientation / normalize
    xVect = normalize(randu(-1,1,3)) #randomVector()
    yVect = normalize(randu(-1,1,3))
    zVect=normalize(np.cross(yVect/norm(yVect),xVect/norm(xVect)))
    xPos=randu(1,6)
    yPos=randu(10,10)
    simID = 0
    for i in range(4): #the 4 allows for the program to use all 90 degree rotation of the object around the y axis

        #a bunch of np to compute the appropriate transformation to achieve the desired orientation
        minX = xVect + E1 #np.add(xVect,[1,0,0])
        rightX = np.cross(xVect,E1)
        minZ = zVect + E3 #np.add(zVect,[0,0,1])
        rightZ=np.cross(zVect,E3)
        vector=normalize(np.cross(np.cross(minX,rightX),np.cross(minZ,rightZ)))
        normVectCrossX=normalize(np.cross(E1,vector))
        xVectCross=normalize(np.cross(xVect,vector))
        normVectCrossZ=normalize(np.cross(E3,vector))
        zVectCross=normalize(np.cross(zVect,vector))
        angle=acos(np.dot(xVectCross,normVectCrossX))
        if np.sum(np.cross(normVectCrossX,xVectCross))/np.sum(vector)<0: # numpy sum has higher precision
            angle=-angle
        angle2=acos(np.dot(zVectCross,normVectCrossZ))
        if np.sum(np.cross(normVectCrossZ,zVectCross))/np.sum(vector)<0:
            angle2=-angle2


       	#runs the simulation for three different z positions
        simulate("R", xPos, yPos, 0, vector[0], vector[1], vector[2], angle, xVect, normalize(np.cross(zVect,xVect)), zVect, speed, scale, objID,trialID,simID)
        simID+=1
        simulate("R", xPos, yPos, 1, vector[0], vector[1], vector[2], angle, xVect, normalize(np.cross(zVect,xVect)), zVect, speed, scale, objID,trialID,simID)
        simID+=1
        simulate("R", xPos, yPos, -1, vector[0], vector[1], vector[2], angle, xVect, normalize(np.cross(zVect,xVect)), zVect, speed, scale, objID,trialID,simID)
        simID+=1

        #rotates the orientation for the next round
        zVect=[0-zVect[1],zVect[0],zVect[2]]
        xVect=[0-xVect[1],xVect[0],xVect[2]]

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

    
    pool = Pool(processes=1,initializer = init, initargs = (counter, ))
    try:
        i = pool.map_async(simulate_obj, trials, chunksize = 1)
        i.wait()
        # print(i.get())
    finally:
        pool.close()
        pool.join()
    
    
