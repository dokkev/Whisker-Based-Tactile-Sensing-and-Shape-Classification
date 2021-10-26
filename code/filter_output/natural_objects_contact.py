import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from numpy.lib.npyio import savetxt
from sklearn.utils.validation import check_array
from read_data import *
from natural_object_class import *


if __name__ == "__main__":

    simID = 0

    np.seterr(invalid='ignore')
    # print("total number of whiskers: ",len(W.whiskers))
    obj_tag = 1 # the object number you want to start with
    obj_max = 2 # the object number you want to end with

    
    while obj_tag < obj_max :
        objFile = objects[obj_tag]

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

            W = NaturalObjects(dirname)
            rownum = len(W.Fx)
            colnum = len(W.Fx[0])-1

            # total number of whiskers counting from 0
            n_max = len(W.whiskers) - 1
            
            # n will direct the specific whisker
            n = 0
            np.set_printoptions(threshold=np.inf)
            # create an empty contact indicator at incident
            contact_indicator,multi_contact = W.indicate_contact(dirname)
            contact_sum = W.sum_contact(np.copy(contact_indicator))
            # print(contact_indicator)

            # np.savetxt("contact_indicator.csv",contact_indicator,delimiter=',')
            # np.savetxt("contact_sum.csv",contact_sum,delimiter=',')
            # np.savetxt("contact_.csv",multi_contact_,delimiter=',')

        
            # convert to image data
            binary_contact_img = convert_contact_to_gray(np.copy(contact_indicator))
            multi_contact_img = convert_contact_to_gray(np.copy(multi_contact))
            contact_sum_img = convert_to_Gray(np.copy(contact_sum))
            # print(np.array(contact_sum_img))

            class_num = W.get_class_num(dirname)

            save_objects_image(dirname,binary_contact_img,class_num,'contact')
            save_objects_image(dirname,multi_contact_img,class_num,'multi_contact')
            save_objects_image(dirname,contact_sum_img,class_num,'contact_sum')


            # class_contact_indicator = W.add_class_num(np.copy(contact_indicator.reshape(5,3375)),dirname)   
            class_contact_sum_indicator = W.add_class_num(np.copy(contact_sum),dirname)   
            
            Total_array2.extend(class_contact_sum_indicator)

       
            trialID += 1

        simID += 1
        obj_tag += 1  

    # save_master('all_contact',Total_array1)
    save_master('contact_sum',Total_array2)
    
