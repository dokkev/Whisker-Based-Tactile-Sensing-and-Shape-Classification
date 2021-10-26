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
    objects_max = 1

    # empyt array for all the data

    np.seterr(invalid='ignore')
    # print("total number of whiskers: ",len(W.whiskers))
    for objID in range(objects_max):
        objFile = objects[objID]

        trialID = 0
        trials_max = 1
        
        
        while trialID < trials_max:

            # # print(trialID)

            # dirname = 'concave24.obj_T010' + '_N02'
            dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
            # dirname = 'overtest'
            print(dirname,"saved")
            # # dirname = 'overtest'
    
            # # default path to dynamic data (each include all whiskers)
            D_dir =  output_dir+(dirname)+'/dynamics/'

            W = WhiskerArray(dirname)
            rownum = len(W.Fx)
            colnum = len(W.Fx[0])-1

            # total number of whiskers counting from 0
            n_max = len(W.whiskers) - 1
            
            # n will direct the specific whisker
            n = 0
            np.set_printoptions(threshold=np.inf)
            # create an empty contact indicator at incident
            contact_indicator,contact_ = W.indicate_contact(dirname)
            contact_sum = W.sum_contact(np.copy(contact_indicator))

        
            # convert to image data
            contact_img = convert_contact_to_gray(np.copy(contact_indicator))
            contact_sum_img = convert_to_Gray(np.copy(contact_sum))
            # print(np.array(contact_sum_img))
            save_image(dirname,contact_img,'contact')
            save_image(dirname,contact_sum_img,'contact_sum')
            
         
           
            W.indicate_protraction()


           

            trialID += 1

        simID += 1
        objID += 1  

    # save_master('all_contact',Total_array1)
    # save_master('all_contact_sum',Total_array2)
    
