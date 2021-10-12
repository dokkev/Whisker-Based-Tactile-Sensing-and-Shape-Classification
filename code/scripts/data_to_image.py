import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib

from sklearn.utils.validation import check_array
from read_data import *


# args
save_master = 0
save_data = 0
classification = 0
save_as_image = 1
indicate_contact = 1

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


if __name__ == "__main__":

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

            # matrix to store data with append and empty it for next trialID
            whisker_fx = []
            whisker_fy = []
            whisker_fz = []
            whisker_mx = []
            whisker_my = []
            whisker_mz = []
            F_img = []
            M_img = []
            # C_img = []

        
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
            contact_indicator = np.zeros((len(Fx),len(Fx[0])-1),dtype=int)
        
          
            count = 1
            if indicate_contact == 1:
                for n in range(len(whiskers)):
                    
                    C = []
                    # whisker name
                    whisker_name = whiskers[n]
            
                    # set target dir with the specific whisker name
                    C_dir = '../output/'+str(dirname)+'/collision/' + str(whisker_name) + '.csv'

                    # get the data from csv file for each whisker
                    C = read_from_csv(C_dir)

                    for x in C:
                        del x[20]

                    for i in range(len(C)):
                        for j in range(len(C[0])):
            
                            C[i][j] = int(C[i][j])

                    if count == 1:
                        C_img = np.array(C)
                    else:
                        C_img = np.hstack((C_img,C))
                       
                   
                    # increment the whisker number
                    count += 1      
                    n += 1

        
            
                # print(C_img)
            for i in range(len(contact_indicator)):
                F = []
                M = []
                for j in range(len(contact_indicator[0])):
                    F_RGB = []
                    M_RGB = []
                    F_RGB.append(float(Fx_array[i][j]))
                    F_RGB.append(float(Fy_array[i][j]))
                    F_RGB.append(float(Fz_array[i][j]))
                    M_RGB.append(float(Mx_array[i][j]))
                    M_RGB.append(float(My_array[i][j]))
                    M_RGB.append(float(Mz_array[i][j]))
                    F.append(F_RGB)
                    M.append(M_RGB)
                F_img.append(F)
                M_img.append(M)
            F_img = np.array(F_img)
            M_img = np.array(M_img)
            F_img[0,:,:] = 0
            M_img[0,:,:] = 0

            rvolume, force_image = convert_to_RGB(F_img, 2)
            rvolume, moment_image = convert_to_RGB(M_img, 2)
            contact_image = convert_contact_to_gray(C_img)
            # rvolume, contact_image = convert_to_RGB(C_img, 19)
            # print(contact_image)

            if save_as_image == 1:
                save_image(dirname,force_image,'f')
                save_image(dirname,moment_image,'m')
                save_image(dirname,contact_image,'c')
            


            trialID += 1
        
        simID += 1
        objID += 1    
