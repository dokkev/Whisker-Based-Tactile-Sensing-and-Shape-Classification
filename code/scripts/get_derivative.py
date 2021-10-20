from re import S
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from numpy.lib.npyio import savetxt

from sklearn.utils.validation import check_array
from read_data import *
from whisker_array import *


# args
indicate_contact = 1


if __name__ == "__main__":

    simID = 0
    objID = 0
    objects_max = 22

    # empyt array for all the data

    np.seterr(invalid='ignore')
    print("total number of whiskers: ",len(whiskers))
    for objID in range(objects_max):
        objFile = objects[objID]

        trialID = 0
        trials_max = 700
        
        
        while trialID < trials_max:

            # # print(trialID)

            # dirname = 'concave24.obj_T010' + '_N02'
            dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
            print(dirname,"saved")
            # # dirname = 'overtest'
    
            # # default path to dynamic data (each include all whiskers)
            D_dir =  '../output/'+(dirname)+'/dynamics/'

            W = WhiskerArray(D_dir,objFile,trialID,simID)
            rownum = len(W.Fx)
            colnum = len(W.Fx[0])-1

            # total number of whiskers counting from 0
            n_max = len(whiskers) - 1
            
            # n will direct the specific whisker
            n = 0

            # create an empty contact indicator at incident
            contact_indicator = W.contact_indicator
           
            # for loop to fill out contact_indicator
            for n in range(len(whiskers)):
    
                # whisker name
                whisker_name = whiskers[n]
        
                # set target dir with the specific whisker name
                C_dir = '../output/'+str(dirname)+'/collision/' + str(whisker_name) + '.csv'

                # get the data from csv file for each whisker
                C = read_from_csv(C_dir)

                # this for loop will take care of the data in row
                for i in range(len(C)):
                    if str(1) in C[i]:
                        contact_indicator[i,n] = int(1)

                # increment the whisker number      
                n += 1
            
            # Set the first row to 0 to prevent error
            contact_indicator[0,:] = 0
            # print numpy arrys all the way
            np.set_printoptions(threshold=np.inf)
           
            # update protraction status
            W.extract_protraction_data(contact_indicator,W.protraction_indicator)
            Fx,Fy,Fz,DFx,DFy,DFz = W.get_mean_and_derivative(W.whisker_fx,W.whisker_fy,W.whisker_fz)
            Mx,My,Mz,DMx,DMy,DMz = W.get_mean_and_derivative(W.whisker_mx,W.whisker_my,W.whisker_mz)

            print(np.shape(Mz))
            # np.savetxt('DMz.csv',DMz,delimiter=',')



            # big 3D array (kind of useless rn)
            Array_5X27X12 = W.Data_to_5_X_27_X_12(Fx,Fy,Fz,DFx,DFy,DFz,Mx,My,Mz,DMx,DMy,DMz)

            # 2D array including 12 data (tabular data)
            Array_5X324 = W.Data_to_5_X_324(Fx,Fy,Fz,DFx,DFy,DFz,Mx,My,Mz,DMx,DMy,DMz)
            # print(np.shape(Array_5X324))

            # 3D array with Mz (image data)
            Array_5X27X2 = W.Data_to_5_X_27_X_2(Mz,DMz)
            # print(np.shape(Array_5X27X2))
            # print(Array_5X27X2)
            # Array_5X27X2[0,:,:] = 0

            # 3D array with Moment xyz (image data)
            Array_5X27X3 = W.Data_to_5_X_27_3(Mx,My,Mz)
        
            # 2D array with Mz (tabular data)
            Array_5X54 = W.Data_to_5_X_54(Mz,DMz)
            # print(np.shape(Array_5X54))

            # 2D array with Average (image data)
            Array_12X27 = W.Data_to_12_X_27(Fx,Fy,Fz,DFx,DFy,DFz,Mx,My,Mz,DMx,DMy,DMz)

            Array_27X12 = Array_12X27.transpose()

            # 1D array with Average (tabular data)
            Array_1X324 = Array_12X27.reshape(1,324)
            # print(np.shape(Array_1X324))

            contact_array = 


            # add concave indication for master tabular data set
            Array_5X324 = W.add_concave_indicator(Array_5X324,dirname)
            Array_1X324 = W.add_concave_indicator(Array_1X324,dirname)
            Array_5X54 = W.add_concave_indicator(Array_5X54,dirname)
            Array_12X27 = W.add_concave_indicator(Array_12X27,dirname)
            Array_27X12 = W.add_concave_indicator(Array_27X12,dirname)

            # Keep tabular data into one master csv
            Total_array1.extend(Array_5X324)
            Total_array2.extend(Array_1X324)
            Total_array3.extend(Array_5X54)
            Total_array4.extend(Array_12X27)
            Total_array5.extend(Array_27X12)

            # Convert 3D matrix to image
            moment_image = convert_to_RGB(Array_5X27X2, 2)
            moment3_image = convert_to_RGB(Array_5X27X3, 2)
            gray_image = convert_to_Gray(Array_12X27)

            # print(np.array(moment_image))
            
            ## Save Images
            # save_image(dirname,moment_image,'mz')
            # save_image(dirname,moment3_image,'mxyz')
            # save_image(dirname,gray_image,'gray')

            trialID += 1

        simID += 1
        objID += 1  

    Total_array1 = np.array(Total_array1)
    Total_array2 = np.array(Total_array2)
    Total_array3 = np.array(Total_array3)
    Total_array4 = np.array(Total_array4)
    
    ## Save Master
    # save_master('data12',Total_array1)
    # save_master('average12',Total_array2)
    # save_master('mz_dmz',Total_array3)
    # save_master('12X27',Total_array4)
    save_master('27X12',Total_array5)