import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from numpy.lib.nanfunctions import nanmean
from numpy.lib.npyio import savetxt
from read_data import *
import warnings
import copy
import pandas as pd
from os import error, walk
from pathlib import Path
import math


# read objects for All whisker simulations
# obj_path = '../data/concave'
# objects = next(walk(obj_path), (None, None, []))[2]  # [] if no file
# objects.sort()

# read objects for rodgers simulations
objects = ['concave40.obj','concave40.obj','concave40.obj',
    'convex40.obj','convex40.obj','convex40.obj']

Total_array1 = []
Total_array2 = []
Total_array3 = []
Total_array4 = []
Total_array5 = []

output_dir = '../output/rodgers_train/'
error_flag = 0
class ConcaveConvex:
    """
    This is Class for concave and convex data. This class should be initialized for processing every simulation.
    """
    def __init__(self,dirname):
        
            # self.dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
            self.dirname = dirname
            D_dir =  output_dir+(dirname)+'/dynamics/'
            my_file = Path(D_dir)
            if my_file.exists():
                error_flag = 0
                # read from csv files
                self.Fx = read_from_csv_2(D_dir,"Fx")
                self.Fy = read_from_csv_2(D_dir,"Fy")
                self.Fz = read_from_csv_2(D_dir,"Fz")
                self.Mx = read_from_csv_2(D_dir,"Mx")
                self.My = read_from_csv_2(D_dir,"My")
                self.Mz = read_from_csv_2(D_dir,"Mz")
                # convert to numpy array
                self.Fx_array = np.array(self.Fx)
                self.Fy_array = np.array(self.Fy)
                self.Fz_array = np.array(self.Fz)
                self.Mx_array = np.array(self.Mx)
                self.My_array = np.array(self.My)
                self.Mz_array = np.array(self.Mz)
            else:
                error_flag = 1
                print("error: cannot load the directory: ",dirname)
    
            # read whisker names
            path = str(output_dir)+str(self.dirname)+'/collision/'
            self.whiskers = next(walk(path), (None, None, []))[2]  # [] if no file
            # soft whisker names in an alphabetical order
            self.whiskers.sort()

            # determine the number of rows and columns
            rownum = int(len(self.Fx))
            colnum = int(len(self.Fx[0])-1)
            div = int(rownum / 125)

            # create empty arrays to store the data
            self.whisker_fx = np.zeros((rownum,colnum))
            self.whisker_fy = np.zeros((rownum,colnum))
            self.whisker_fz = np.zeros((rownum,colnum))
            self.whisker_mx = np.zeros((rownum,colnum))
            self.whisker_my = np.zeros((rownum,colnum))
            self.whisker_mz = np.zeros((rownum,colnum))

            self.contact_indicator = np.zeros((rownum,colnum),dtype=int)

    def pick_protraction_moment(self,My,Mz):
        # pick the protraction moment
        my = np.copy(My)
        mz = np.copy(Mz)
     

        M_peak = []


        for i in range(len(my)):
            for j in range(len(my[0])-1):
                if i == 62:
                    m = ((float(mz[i,j])**2) + float(my[i,j])**2)**0.5
                    M_peak.append(m)
        M_peak = np.array(M_peak)

        return M_peak

      



    

    def indicate_contact(self,dirname):
        # for loop to fill out contact_indicator
        contact_ = []
        contact_indicator =np.copy(self.contact_indicator)
        for n in range(len(self.whiskers)):
            
            # whisker name
            whisker_name = self.whiskers[n]

            # set target dir with the specific whisker name
            C_dir = str(output_dir)+str(dirname)+'/collision/' + str(whisker_name)

            # get the data from csv file for each whisker
            C = read_from_csv(C_dir)
            C_array = np.delete(np.array(C),20,axis=1)
            C_array = C_array.astype(np.int)
            sum = np.sum(C_array,axis=1)
            sum.tolist()
            contact_.append(sum)

            
            # this for loop will take care of the data in row
            for i in range(len(C)):
                if str(1) in C[i]:
                    contact_indicator[i,n] = int(1)
                  
            
            # increment the whisker number      
            n += 1
        # Set the first row to 0 to prevent error
        contact_indicator[0,:] = 0
        binary_contact_indicator = np.array(contact_indicator)

        multi_contact_indicator = np.array(contact_)
        multi_contact_indicator = np.transpose(contact_)

        return binary_contact_indicator,multi_contact_indicator

    def sum_contact(self,contact_indicator):
        c1,c2,c3,c4,c5 = np.array_split(contact_indicator,5)

        c1 = np.sum(c1,axis=0)
        c2 = np.sum(c2,axis=0)
        c3 = np.sum(c3,axis=0)
        c4 = np.sum(c4,axis=0)
        c5 = np.sum(c5,axis=0)

        contact_sum = np.vstack((c1,c2,c3,c4,c5))

        return contact_sum


    def separate_contact(self,contact_indicator):
        c1,c2,c3,c4,c5 = np.array_split(contact_indicator,5)

        c1 = np.sum(c1,axis=0)
        c2 = np.sum(c2,axis=0)
        c3 = np.sum(c3,axis=0)
        c4 = np.sum(c4,axis=0)
        c5 = np.sum(c5,axis=0)

        return c1,c2,c3,c4,c5

    
    def extract_protraction_data(self,contact_indicator,protraction_indicator):
        for i in range(len(contact_indicator)):
                for j in range(len(contact_indicator[0])):
                    if int(contact_indicator[i,j]) == int(1) and int(protraction_indicator[i]) == int(1):
                        self.whisker_fx[i,j] = float(self.Fx_array[i,j])
                        self.whisker_fy[i,j] = float(self.Fy_array[i,j])
                        self.whisker_fz[i,j] = float(self.Fz_array[i,j])
                        self.whisker_mx[i,j] = float(self.Mx_array[i,j])
                        self.whisker_my[i,j] = float(self.My_array[i,j])
                        self.whisker_mz[i,j] = float(self.Mz_array[i,j])
                    else:
                        self.whisker_fx[i,j] = np.nan
                        self.whisker_fy[i,j] = np.nan
                        self.whisker_fz[i,j] = np.nan
                        self.whisker_mx[i,j] = np.nan
                        self.whisker_my[i,j] = np.nan
                        self.whisker_mz[i,j] = np.nan
        # replace 0 to nan
        self.whisker_fx[self.whisker_fx==0]=['nan']
        self.whisker_fy[self.whisker_fy==0]=['nan']
        self.whisker_fz[self.whisker_fz==0]=['nan']
        self.whisker_mx[self.whisker_mx==0]=['nan']
        self.whisker_my[self.whisker_my==0]=['nan']
        self.whisker_mz[self.whisker_mz==0]=['nan']


    def get_mean_and_derivative(self,data_x,data_y,data_z):
        """
        This function will calculate the mean and derivative of the data for each cycle of whisking (125 time steps)
        For my simulation, I set the cycle to be 5 which has 625 time steps.

        """
    
        x1,x2,x3,x4,x5 = np.array_split(data_x,5)
        y1,y2,y3,y4,y5 = np.array_split(data_y,5)
        z1,z2,z3,z4,z5 = np.array_split(data_z,5)


        # get mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x1 = np.nanmean(x1,axis=0)
            x2 = np.nanmean(x2,axis=0)
            x3 = np.nanmean(x3,axis=0)
            x4 = np.nanmean(x4,axis=0)
            x5 = np.nanmean(x5,axis=0)
            y1 = np.nanmean(y1,axis=0)
            y2 = np.nanmean(y2,axis=0)
            y3 = np.nanmean(y3,axis=0)
            y4 = np.nanmean(y4,axis=0)
            y5 = np.nanmean(y5,axis=0)
            z1 = np.nanmean(z1,axis=0)
            z2 = np.nanmean(z2,axis=0)
            z3 = np.nanmean(z3,axis=0)
            z4 = np.nanmean(z4,axis=0)
            z5 = np.nanmean(z5,axis=0)
        
        # get derivative (each time step is 0.001 sec)
        dx1 = np.divide(x1,(np.count_nonzero(~np.isnan(x1),axis=0) * 0.001))
        dx2 = np.divide(x2,(np.count_nonzero(~np.isnan(x2),axis=0) * 0.001))
        dx3 = np.divide(x3,(np.count_nonzero(~np.isnan(x3),axis=0) * 0.001))
        dx4 = np.divide(x4,(np.count_nonzero(~np.isnan(x4),axis=0) * 0.001))
        dx5 = np.divide(x5,(np.count_nonzero(~np.isnan(x5),axis=0) * 0.001))
        dy1 = np.divide(y1,(np.count_nonzero(~np.isnan(y1),axis=0) * 0.001))
        dy2 = np.divide(y2,(np.count_nonzero(~np.isnan(y2),axis=0) * 0.001))
        dy3 = np.divide(y3,(np.count_nonzero(~np.isnan(y3),axis=0) * 0.001))
        dy4 = np.divide(y4,(np.count_nonzero(~np.isnan(y4),axis=0) * 0.001))
        dy5 = np.divide(y5,(np.count_nonzero(~np.isnan(y5),axis=0) * 0.001))
        dz1 = np.divide(z1,(np.count_nonzero(~np.isnan(z1),axis=0) * 0.001))
        dz2 = np.divide(z2,(np.count_nonzero(~np.isnan(z2),axis=0) * 0.001))
        dz3 = np.divide(z3,(np.count_nonzero(~np.isnan(z3),axis=0) * 0.001))
        dz4 = np.divide(z4,(np.count_nonzero(~np.isnan(z4),axis=0) * 0.001))
        dz5 = np.divide(z5,(np.count_nonzero(~np.isnan(z5),axis=0) * 0.001))

        # combine divided arrays again
        x = np.vstack((x1,x2,x3,x4,x5))
        y = np.vstack((y1,y2,y3,y4,y5))
        z = np.vstack((z1,z2,z3,z4,z5))
        dx = np.vstack((dx1,dx2,dx3,dx4,dx5))
        dy = np.vstack((dy1,dy2,dy3,dy4,dy5))
        dz = np.vstack((dz1,dz2,dz3,dz4,dz5))
    
        return x,y,z,dx,dy,dz
        
            
    def indicate_protraction(self,rownum):
        """
        This function creates a N X 1 array that indicates protraction & retraction status
        """
        # empty protraction indicator
        protraction_indicator = np.zeros((rownum,1),dtype=int)

        # 1=protraction, 0=retraction, 2=stationary
        # create protraction indicator cycles every 125 frames
        protraction_counter = 1
        for i in range(len(protraction_indicator)):
            if int(protraction_counter) < int(63):
                protraction_indicator[i] = 1 # protraction
            elif int(protraction_counter) == int(63):
                protraction_indicator[i] = 2 # stationary
            elif int(63) < int(protraction_counter) < int(125):
                protraction_indicator[i] = 0 # retraction
            elif int(protraction_counter) == int(125):
                protraction_indicator[i] = 2 # stationary

            protraction_counter += int(1)
            if int(protraction_counter) == int(126):
                protraction_counter = 1

        return protraction_indicator

    def add_concave_indicator(self,data,dirname):
            if str("concave") in dirname:
                concave_indicator = np.full((len(data),1),int(1))
                # print(concave_indicator)
                data = np.hstack((data,concave_indicator))
            elif str("convex") in dirname:
                convex_indicator = np.zeros((len(data),1),dtype=int)
                # print(convex_indicator)
                data = np.hstack((data,convex_indicator))
            else:
                print("Dirname: *%s*  does not include info about object type"% dirname )
        

            return data

    def add_convace_indicator_flat(self,data,dirname):
            if str("concave") in dirname:
                concave_indicator = 1

                data = np.append(data,concave_indicator)
            elif str("convex") in dirname:
                convex_indicator = 0
      
                data = np.append(data,convex_indicator)
            else:
                print("Dirname: *%s*  does not include info about object type"% dirname )
        
            return data