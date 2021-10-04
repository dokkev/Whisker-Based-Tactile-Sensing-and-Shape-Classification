import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from sklearn.preprocessing import normalize
from read_kinematics import *

"""
From each whisker's xyz position and dynamic data, let's create a tacile 3D mapping 
Each whisker has 20 segments, and only segments expierenced collsion will be mapped.

"""
# Fixing random state for reproducibility
np.random.seed(19680801)



# whiskers names array
whiskers = [
            "RA0","RA1","RA2","RA3","RA4",
            "RB0","RB1","RB2","RB3","RB4",
            "RC0","RC1","RC2","RC3","RC4","RC5",
            "RD0","RD1","RD2","RD3","RD4","RD5",
            "RE1","RE2","RE3","RE4","RE5"]

# whiskers=["RC0","RC1","RC2","RC3"]

# name of the dir in output
trials = 0
trials_max = 4
for trials in range(trials_max):

    dirname = "kine"

    # Let's plot and whisker position for each whisker consecutively
    # initialize the figure
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # uax = fig.gca(projection='3d')

    # matrix to store data with append
    whisker_x = []
    whisker_y = []
    whisker_z = []
    whisker_fx = []
    whisker_fy = []
    whisker_fz = []
    whisker_mx = []
    whisker_my = []
    whisker_mz = []


    # default path to dynamic data (each include all whiskers)
    D_dir =  '../output/'+str(dirname)+'/dynamics/'
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
    # print("total number of whiskers: ",len(whiskers))

    # n will direct the specific whisker
    n = 0
    while n <= n_max:
        
        # whisker name
        whisker_name = whiskers[n]
    
        # set target dir with the specific whisker name
        C_dir = '../output/'+str(dirname)+'/collision/' + str(whisker_name) + '.csv'
        X_dir = '../output/'+str(dirname)+'/kinematics/x/' + str(whisker_name) + '.csv'
        Y_dir = '../output/'+str(dirname)+'/kinematics/y/' + str(whisker_name) + '.csv'
        Z_dir = '../output/'+str(dirname)+'/kinematics/x/' + str(whisker_name) + '.csv'
        

        # get the data from csv file for each whisker
        C = read_from_csv(C_dir)
        X = read_from_csv(X_dir)
        Y = read_from_csv(Y_dir)
        Z = read_from_csv(Z_dir)

        # print(len(C[0]))
        # this for loop will take care of the data in row
        for i in range(len(C)):
            if str(1) in C[i]:
                # Fx_array = np.delete(Fx_array,i,0)
        
                whisker_fx.append(float(Fx[i][n]))
                whisker_fy.append(float(Fy[i][n]))
                whisker_fz.append(float(Fz[i][n]))
                whisker_mx.append(float(Mx[i][n]))
                whisker_my.append(float(My[i][n]))
                whisker_mz.append(float(Mz[i][n]))
                # print("collsion!")
    

            # this for loop takes care of the column of each row
            for j in range(len(C[0])):
                
                # if the collision data is 1, then we will plot the position. Make sure that you are comapring to string
                if str(C[i][j]) == str(1):
                    # plot the position
                    # print("Collision on: ",whisker_name)
                    whisker_x.append(float(X[i][j]))
                    whisker_y.append(float(Y[i][j]))
                    whisker_z.append(float(Z[i][j]))

                    
                    
        # increment the whisker number      
        n += 1

    # combine into a single array
    whisker_pos = np.array([whisker_x,whisker_y,whisker_z])
    whisker_f = np.array([whisker_fx,whisker_fy,whisker_fz])
    whisker_m = np.array([whisker_mx,whisker_my,whisker_mz])

    # save data
    save_data(dirname,whisker_pos,"pos")
    # save_data(dirname,whisker_f,"f")
    # save_data(dirname,whisker_m,"m")
    print("saved")

    # normalize the data
    # whisker_pos = normalize(whisker_pos)
    # whisker_f = normalize(whisker_f)
    # whisker_m = normalize(whisker_m)


    # 3D Vector Plot
    # dynamic_data = whisker_f
    # u,v,w = vector_plot_3d(dynamic_data[0],dynamic_data[1],dynamic_data[2])
    # uax.quiver(dynamic_data[0], dynamic_data[1], dynamic_data[2], u, v, w, length=0.1, normalize=True)

    # ax.scatter(whisker_pos[0],whisker_pos[1],whisker_pos[2],marker='o',color='r')
    # ax.scatter(whisker_f[0],whisker_f[1],whisker_f[2],marker='o',color='b')
    # ax.scatter(whisker_m[0],whisker_m[1],whisker_m[2],marker='o',color='g')
    # plt.show()
    # print("yes!")

trials += 1

    


