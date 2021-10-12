import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from sklearn.preprocessing import normalize
from read_data import *
from PIL import Image

"""
This script extract dynamic data which experienced collision with the object at its time step in the simulation.
It wirtes csv files of each tiral for each object along with a big summary csv file includes all the trials of same types of objects.

"""

# args
save_master = 0
save_data = 1
classification = 1


# whiskers names array
whiskers = [
            "RA0","RA1","RA2","RA3","RA4",
            "RB0","RB1","RB2","RB3","RB4",
            "RC0","RC1","RC2","RC3","RC4","RC5",
            "RD0","RD1","RD2","RD3","RD4","RD5",
            "RE1","RE2","RE3","RE4","RE5"]

objects = [ 'concave20.obj','concave22.obj','concave24.obj','concave26.obj','concave28.obj',
            'concave30.obj','concave32.obj','concave34.obj','concave36.obj','concave38.obj',
            'concave40.obj',
            'convex20.obj','convex22.obj','convex24.obj','convex26.obj','convex28.obj',
            'convex30.obj','convex32.obj','convex34.obj','convex36.obj','convex38.obj',
            'convex40.obj']

# whiskers=["RC0","RC1","RC2","RC3"]

# name of the dir in output
simID = 0
objID = 0
objects_max = 22

# There are "ultimate" arrays to keep ALL concave data in one single file for all simulations

F_concave = []
M_concave = []
F_convex = []
M_convex = []
F_total = []
M_total = []

print("total number of whiskers: ",len(whiskers))
for objID in range(objects_max):
    objFile = objects[objID]

    trialID = 0
    trials_max = 100
    
    
    # print("===== NEXT OBJECT ====")
    

    while trialID < trials_max:
        # print(trialID)

        # dirname = 'concave24.obj_T010' + '_N02'
        dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
        print(dirname,"saved")
        # dirname = 'overtest'

        # matrix to store data with append and empty it for next trialID
        whisker_fx = []
        whisker_fy = []
        whisker_fz = []
        whisker_mx = []
        whisker_my = []
        whisker_mz = []
        Img = []
    

        # default path to dynamic data (each include all whiskers)
        D_dir =  '../output/'+(dirname)+'/dynamics/'
        # read the dynamic data
        Fx = read_from_csv_2(D_dir,"Fx")
        Fy = read_from_csv_2(D_dir,"Fy")
        Fz = read_from_csv_2(D_dir,"Fz")
        Mx = read_from_csv_2(D_dir,"Mx")
        My = read_from_csv_2(D_dir,"My")
        Mz = read_from_csv_2(D_dir,"Mz")
        Fx_array = np.array(Fx)
        Fy_array = np.array(Fy)
        Fz_array = np.array(Fz)
        Mx_array = np.array(Mx)
        My_array = np.array(My)
        Mz_array = np.array(Mz)

        # total number of whiskers counting from 0
        n_max = len(whiskers) - 1
        
        # n will direct the specific whisker
        n = 0

        # create an empty contact indicator at incident
        contact_indicator = np.zeros((len(Fx),len(Fx[0])-1),dtype=int)
        # print(contact_indicator)
        # np.savetxt("contact_indicator.csv",contact_indicator,delimiter=',')

        for n in range(len(whiskers)):
            # 1 X 3 Matrix to hold rgb values
            F_RGB = []
            M_RGB = []


            # whisker name
            whisker_name = whiskers[n]
    
            # set target dir with the specific whisker name
            C_dir = '../output/'+str(dirname)+'/collision/' + str(whisker_name) + '.csv'

            # get the data from csv file for each whisker
            C = read_from_csv(C_dir)

            # this for loop will take care of the data in row
            # print(len(C)-1)
            for i in range(len(C)):
                if str(1) in C[i]:
                    contact_indicator[i,n] = 1

            # set the first row of the contact indicator to 0
            contact_indicator[0,:] = 0



            # increment the whisker number      
            n += 1

  
        # np.savetxt("contact_indicator.csv",contact_indicator,delimiter=',')   
        # print(contact_indicator)  
        # print(len(contact_indicator)) # make sure the length of the array is corresponding to the data size
        # print(len(Fx[0]))
        # Let's compare the dynamic data to the contact_indicator
        for i in range(len(contact_indicator)):
            for j in range(len(contact_indicator[0])):
                if int(contact_indicator[i,j]) == int(1):
                    whisker_fx.append(float(Fx_array[i][j]))
                    whisker_fy.append(float(Fy_array[i][j]))
                    whisker_fz.append(float(Fz_array[i][j]))
                    whisker_mx.append(float(Mx_array[i][j]))
                    whisker_my.append(float(My_array[i][j]))
                    whisker_mz.append(float(Mz_array[i][j]))
                             
        # print(len(whisker_fx))
    
 
        # combine all the data into one array
        whisker_f = np.array([whisker_fx,whisker_fy,whisker_fz]).transpose()
        whisker_m = np.array([whisker_mx,whisker_my,whisker_mz]).transpose()


        # add concave / concvex indicator
        if classification == 1:
            if str("concave") in dirname:
                concave_indicator = np.full((len(whisker_f),1),int(1))
                # print(concave_indicator)
                whisker_f = np.hstack((whisker_f,concave_indicator))
                whisker_m = np.hstack((whisker_m,concave_indicator))

            elif str("convex") in dirname:
                convex_indicator = np.zeros((len(whisker_f),1),dtype=int)
                whisker_f = np.hstack((whisker_f,convex_indicator))
                whisker_m = np.hstack((whisker_m,convex_indicator))


        # save data
        if save_data == 1:
            save_data(dirname,whisker_f,"f")
            save_data(dirname,whisker_m,"m")
        # print("trial number",trialID,"saved")

        if save_master == 1:
            if str("concave") in dirname:
                # print("This is Concave!")
                F_concave.extend(whisker_f)
                M_concave.extend(whisker_m)
                # print(len(F_concave))

            elif str("convex") in dirname:
                # print("This is Convex!")
                F_convex.extend(whisker_f)
                M_convex.extend(whisker_m)

            F_total.extend(whisker_f)
            M_total.extend(whisker_m)


        trialID += 1

    simID += 1
    objID += 1    


if save_master == 1:
    # conver all master arrys to numpy array
    F_concave = np.array(F_concave)
    M_concave = np.array(M_concave)
    F_convex = np.array(F_convex)
    M_convex = np.array(M_convex)
    F_total = np.array(F_total)
    M_total = np.array(M_total)
    save_master("F_concave_total",F_concave)
    save_master("M_concave_total",M_concave)
    save_master("F_convex_total",F_concave)
    save_master("M_convex_total",M_convex)
    save_master("F_total",F_total)
    save_master("M_total",M_total)

print("ALL SAVED!")

