import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from numpy.lib.npyio import savetxt
from sklearn.utils.validation import check_array
from read_data import *
from whisker_array import *


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
            np.set_printoptions(threshold=np.inf)
            # create an empty contact indicator at incident
            contact_indicator = W.indicate_contact(dirname,whiskers)
            contact_sum = W.sum_contact(np.copy(contact_indicator))
            # print(contact_sum)
        
            # convert to image data
            contact_img = convert_contact_to_gray(np.copy(contact_indicator))
            contact_sum_img = convert_to_Gray(np.copy(contact_sum))
            # print(np.array(contact_sum_img))
            save_image(dirname,contact_img,'contact')
            save_image(dirname,contact_sum_img,'contact_sum')
            
         

            concave_contact_indicator = W.add_concave_indicator(np.copy(contact_indicator.reshape(5,3375)),dirname)   
            concave_contact_sum_indicator = W.add_concave_indicator(np.copy(contact_sum),dirname)   
           
           
            W.indicate_protraction()
            # np.savetxt('protraction_indicator.csv', W.protraction_indicator, delimiter=",")

            # contact_protraction_indicator = contact_indicator
            # for i in range(len(contact_indicator)):
            #     for j in range(len(contact_indicator[i])):
            #         if int(W.protraction_indicator[i]) == int(0):
            #             contact_protraction_indicator[i][j] = 0
            #         else:
            #             pass
                

            # 5 X 3375 contact array
            Total_array1.extend(concave_contact_indicator)
            Total_array2.extend(concave_contact_sum_indicator)

           

            trialID += 1

        simID += 1
        objID += 1  

    save_master('all_contact',Total_array1)
    save_master('all_contact_sum',Total_array2)
    
