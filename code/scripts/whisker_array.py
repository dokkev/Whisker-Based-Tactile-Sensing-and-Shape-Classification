import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from numpy.lib.nanfunctions import nanmean
from numpy.lib.npyio import savetxt
from read_data import *
import warnings

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

Total_array1 = []
Total_array2 = []
Total_array3 = []
Total_array4 = []


class WhiskerArray:
    def __init__(self,D_dir,objFile,trialID,simID):
            self.dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
            self.Fx = read_from_csv_2(D_dir,"Fx")
            self.Fy = read_from_csv_2(D_dir,"Fy")
            self.Fz = read_from_csv_2(D_dir,"Fz")
            self.Mx = read_from_csv_2(D_dir,"Mx")
            self.My = read_from_csv_2(D_dir,"My")
            self.Mz = read_from_csv_2(D_dir,"Mz")

            self.Fx_array = np.array(self.Fx)
            self.Fy_array = np.array(self.Fy)
            self.Fz_array = np.array(self.Fz)
            self.Mx_array = np.array(self.Mx)
            self.My_array = np.array(self.My)
            self.Mz_array = np.array(self.Mz)

            
            rownum = int(len(self.Fx))
            colnum = int(len(self.Fx[0])-1)
            div = int(rownum / 125)
            new_row = int(rownum / div)
       
            self.whisker_fx = np.zeros((rownum,colnum))
            self.whisker_fy = np.zeros((rownum,colnum))
            self.whisker_fz = np.zeros((rownum,colnum))
            self.whisker_mx = np.zeros((rownum,colnum))
            self.whisker_my = np.zeros((rownum,colnum))
            self.whisker_mz = np.zeros((rownum,colnum))

            self.protraction_indicator = np.zeros((rownum,1),dtype=int)
            self.indicate_protraction()

            self.contact_indicator = np.zeros((rownum,colnum),dtype=int)


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
        
        # get derivative
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
        
    def Data_to_5_X_27_X_12(self,fx,fy,fz,dfx,dfy,dfz,mx,my,mz,dmx,dmy,dmz):
        Array_5X27X12 =[]
        for i in range(len(fx)):
            A_comb = [] # Combination of subarray
            for j in range(len(fx[0])):
                A12 = [] #Sub-array contains 12 data
                A12.append(fx[i,j])
                A12.append(fy[i,j])
                A12.append(fz[i,j])
                A12.append(dfx[i,j])
                A12.append(dfy[i,j])
                A12.append(dfz[i,j])
                A12.append(mx[i,j])
                A12.append(my[i,j])
                A12.append(mz[i,j])
                A12.append(dmx[i,j])
                A12.append(dmy[i,j])
                A12.append(dmz[i,j])
                A_comb.append(A12)
            Array_5X27X12.append(A_comb)

        return_array = np.array(Array_5X27X12)
        return_array = np.nan_to_num(return_array)

        return return_array
    


    def Data_to_5_X_324(self,fx,fy,fz,dfx,dfy,dfz,mx,my,mz,dmx,dmy,dmz):
        return_array = []
        for i in range(len(fx)):
            A12 = []
            for j in range(len(fx[0])):
                A12.append(fx[i,j])
                A12.append(fy[i,j])
                A12.append(fz[i,j])
                A12.append(dfx[i,j])
                A12.append(dfy[i,j])
                A12.append(dfz[i,j])
                A12.append(mx[i,j])
                A12.append(my[i,j])
                A12.append(mz[i,j])
                A12.append(dmx[i,j])
                A12.append(dmy[i,j])
                A12.append(dmz[i,j])
            return_array.append(A12)
    
        return_array = np.array(return_array)
        return_array = np.nan_to_num(return_array)

        return return_array



    def Data_to_12_X_27(self,fx,fy,fz,dfx,dfy,dfz,mx,my,mz,dmx,dmy,dmz):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fx = np.nanmean(fx,axis=0)
            fy = np.nanmean(fy,axis=0)
            fz = np.nanmean(fz,axis=0)
            dfx = np.nanmean(dfx,axis=0)
            dfy = np.nanmean(dfy,axis=0)
            dfz = np.nanmean(dfz,axis=0)
            mx = np.nanmean(mx,axis=0)
            my = np.nanmean(my,axis=0)
            mz = np.nanmean(mz,axis=0)
            dmx = np.nanmean(dmx,axis=0)
            dmy = np.nanmean(dmy,axis=0)
            dmz = np.nanmean(dmz,axis=0)

        Array_12X27 =[]
        for i in range(len(fx)):
            A_12 = [] # Combination of subarray
            A12 = [] #Sub-array contains 12 data
            A12.append(fx[i])
            A12.append(fy[i])
            A12.append(fz[i])
            A12.append(dfx[i])
            A12.append(dfy[i])
            A12.append(dfz[i])
            A12.append(mx[i])
            A12.append(my[i])
            A12.append(mz[i])
            A12.append(dmx[i])
            A12.append(dmy[i])
            A12.append(dmz[i])
            Array_12X27.append(A12)
    
        
        return_array = np.array(Array_12X27)
        return_array = np.transpose(return_array)
        # print(np.shape(return_array))
        return_array = np.nan_to_num(return_array)

        return return_array
    


    def Data_to_5_X_27_X_2(self,m,dm):
        array_5X27X2 = []
        for i in range(len(m)):
            A_comb = [] # Combination of subarray
            for j in range(len(m[0])):
                A2 = []
                A2.append(m[i,j])
                A2.append(dm[i,j])
                A2.append(0)
                A_comb.append(A2)
            array_5X27X2.append(A_comb)

        return_array = np.array(array_5X27X2)
        return_array = np.nan_to_num(return_array)
    
        return return_array

    def Data_to_5_X_27_3(self,mx,my,mz):
        array_5X27X3 = []
        for i in range(len(mx)):
            A_comb = [] # Combination of subarray
            for j in range(len(mx[0])):
                A2 = []
                A2.append(mx[i,j])
                A2.append(my[i,j])
                A2.append(mz[i,j])
                A_comb.append(A2)
            array_5X27X3.append(A_comb)

        return_array = np.array(array_5X27X3)
        return_array = np.nan_to_num(return_array)

        return return_array


    def Data_to_5_X_54(self,m,dm):
        return_array = []
        for i in range(len(m)):
            A2 = []
            for j in range(len(m[0])):
                A2.append(m[i,j])
                A2.append(dm[i,j])
            return_array.append(A2)

        return_array = np.array(return_array)
        return_array = np.nan_to_num(return_array)
    
        return return_array


                
    
    def indicate_protraction(self):
        # empty protraction indicator
        protraction_indicator = self.protraction_indicator

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
           
                        
               
    def add_concave_indicator(self,data,dirname):
            if str("concave") in dirname:
                concave_indicator = np.full((len(data),1),int(1))
                # print(concave_indicator)
                data = np.hstack((data,concave_indicator))
            elif str("convex") in dirname:
                convex_indicator = np.zeros((len(data),1),dtype=int)
                # print(convex_indicator)
                data = np.hstack((data,convex_indicator))

            return data