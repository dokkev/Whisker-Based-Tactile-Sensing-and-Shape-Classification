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
from numpy.lib.npyio import savetxt

counter = None

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
def simulate(whisker, ObjX, ObjY, ObjZ, ObjYAW, ObjPITCH, ObjROLL,objID,trialID,simID,objName):
    global counter
    #with counter.get_lock():
    #    counter.value += 1
    
    objects = ['scan_00.obj','scan_01.obj','scan_02.obj','scan_03.obj','scan_04.obj','scan_05.obj', \
        'scan_06.obj','scan_07.obj','scan_08.obj','scan_09.obj','scan_10.obj','scan_11.obj','scan_12.obj',\
        'scan_13.obj','scan_14.obj','scan_15.obj','scan_16.obj','scan_17.obj','scan_18.obj','scan_19.obj',\
        'scan_20.obj','scan_21.obj','scan_22.obj','scan_23.obj','scan_24.obj','scan_25.obj','scan_26.obj',\
        'scan_27.obj','scan_28.obj','scan_29.obj','scan_30.obj','scan_31.obj','scan_32.obj','scan_33.obj',\
        'scan_34.obj','scan_35.obj','scan_36.obj','scan_37.obj',\
        'scan_38.obj','scan_39.obj','scan_40.obj','scan_41.obj','scan_42.obj','scan_43.obj','scan_44.obj',\
        'scan_45.obj','scan_46.obj','scan_47.obj','scan_48.obj','scan_49.obj','scan_50.obj','scan_51.obj',\
        'scan_52.obj','scan_53.obj','scan_54.obj','scan_55.obj','scan_56.obj','scan_57.obj','scan_58.obj',\
        'scan_59.obj','scan_60.obj','scan_61.obj','scan_62.obj','scan_63.obj','scan_64.obj','scan_65.obj',\
        'scan_66.obj','scan_67.obj','scan_68.obj','scan_69.obj','scan_70.obj','scan_71.obj','scan_72.obj',\
        'scan_73.obj','scan_74.obj','scan_75.obj','scan_76.obj','scan_77.obj','scan_78.obj','scan_79.obj',\
        'scan_80.obj','scan_81.obj','scan_82.obj','scan_83.obj','scan_84.obj','scan_85.obj','scan_86.obj',\
        'scan_87.obj','scan_88.obj','scan_89.obj','scan_90.obj','scan_91.obj','scan_92.obj','scan_93.obj',\
        'scan_94.obj','scan_95.obj','scan_96.obj','scan_97.obj']
    
    # np.savetxt('object_param.csv',objects,delimiter=',')
    
    # print('Simulating: ' + str(whisker))
    objFile = objects[objID]

    filename =  str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d') #curr_time.replace(":","-")
    dirout = "../output/natural_objects/"+filename
    
    # print(dirout)
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)

    str1 = "../build/whiskit \
    --PRINT 2 \
    --CDIST 50 \
    --SIM_TIME 0.625 \
    --SAVE_KINEMATICS 0 \
    --WHISKER_NAMES R \
    --CPITCH 0 \
    --CYAW 180 \
    --BLOW 1  \
    --OBJECT 5 \
    --ACTIVE 1 \
    --SAVE_VIDEO 0 \
    --SAVE 1 "print("Simulation Object: " + str(objFile))

    str2 = " --file_env ../data/natural_objects/" + objFile
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
    print("Simulation Object: " + str(objFile))
    print("Object Name: " + str(objName))
    print("Now Whisking: " + str(objFile))
    print("X = " + str(ObjX) + " Y = " + str(ObjY) + " Z = " +str(ObjZ))
    print("YAW = " + str(ObjYAW) + " PITCH = " + str(ObjPITCH) + " ROLL = " +str(ObjROLL))
    print("ended whiskit")
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
    obj_tag = 1 # the object number you want to start with
    obj_max = 98 # the object number you want to end with

    
    while obj_tag < obj_max :
    
        # open the object parameters
        
        with open('obj_param.csv','r')as file:
            filecontent=csv.reader(file)
            line_j = list(filecontent)
            row = line_j[obj_tag]
            obj_num = int(row[1])
            obj_name = row[2]
            x = round(float(row[4]) + random.uniform(-5.0, 5.0),4)
            y = round(float(row[5]) + random.uniform(-5.0, 5.0),4)
            z = round(float(row[6]) + random.uniform(-5.0, 5.0),4)
            yaw = round(float(row[7]) + random.uniform (-0.3,0.3),4)
            pitch = round(float(row[8]) + random.uniform (-0.3,0.3),4)
            roll = round(float(row[9]) + random.uniform (-0.3,0.3),4)

        simulate("R", x, y, z, yaw, pitch, roll, obj_num, trialID, simID,obj_name)
        simID+=1
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
    numConfig = 1000 # how many times you want to simulate
    trials = []
    for n in range(numConfig):
        trials.append([2,trialbase+n])


    pool = Pool(processes=54,initializer = init, initargs = (counter, ))
    try:
        i = pool.map_async(simulate_obj, trials, chunksize = 1)
        i.wait()
    finally:
        pool.close()
        pool.join()
    
    
    
