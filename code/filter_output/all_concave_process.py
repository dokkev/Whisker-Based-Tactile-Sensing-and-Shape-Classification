import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pathlib
from numpy.lib.npyio import save, savetxt
from sklearn.utils.validation import check_array
from read_data import *
from concave_convex_class import *

"""
This script process data from concave-convex simulation to produce training data for the tabular and image classifiers.
It is specifically designed to process concave-convex simulation with ALL whiskers.
This script wasn't designed to be compatible with other simulations such as natural objects simulation.
"""

# Settings
np.set_printoptions(threshold=np.inf)


if __name__ == "__main__":

    simID = 0
    objID = 0
    objects_max = 21 # number of objects to process

    for objID in range(objects_max):
        objFile = objects[objID]

        trialID = 0    # inital number of trials to process
        trials_max = 300 # final number of trials to process
        
        while trialID < trials_max:

            # determine dirname which should be in the same format as dir in output
            dirname = str(objFile) + '_T' + format(trialID, '03d') + '_N' + format(simID, '02d')
            # dirname = 'alltest'
           
            # Initialize the class
            W = ConcaveConvex(dirname)
            rownum = len(W.Fx)
            colnum = len(W.Fx[0])-1

    

            # create a protraction indicator
            protraction_indicator = W.indicate_protraction(rownum)
            # create a contact indicator
            contact_indicator,multi_contact_indicator = W.indicate_contact(dirname)
            contact_sum = W.sum_contact(contact_indicator)
            c1,c2,c3,c4,c5 = W.separate_contact(contact_indicator)
            # extract data based on contact & protraction
            W.extract_protraction_data(contact_indicator,protraction_indicator)
     
            # get average and deviation of the processed data
            # Fx,Fy,Fz,DFx,DFy,DFz = W.get_mean_and_derivative(W.whisker_fx,W.whisker_fy,W.whisker_fz)
            # Mx,My,Mz,DMx,DMy,DMz = W.get_mean_and_derivative(W.whisker_mx,W.whisker_my,W.whisker_mz)
            
            # Mz = np.nan_to_num(Mz)
            # Mz = W.add_concave_indicator(Mz,dirname)

            # contact_sum = W.add_concave_indicator(contact_sum,dirname)

            # Total_array1.extend(Mz)
            # Total_array2.extend(contact_sum)

            # convert to image data
            c1 = convert_to_Gray(c1)
            c2 = convert_to_Gray(c2)
            c3 = convert_to_Gray(c3)
            c4 = convert_to_Gray(c4)
            c5 = convert_to_Gray(c5)
            # contact_sum = convert_to_Gray(contact_sum)

            # save the image data
            save_image_5sections("c1",dirname,c1,"ALL")
            save_image_5sections("c2",dirname,c2,"ALL")
            save_image_5sections("c3",dirname,c3,"ALL")
            save_image_5sections("c4",dirname,c4,"ALL")
            save_image_5sections("c5",dirname,c5,"ALL")
            # save_image(dirname,contact_sum,"ALL")
            

            print(dirname,"saved")
                
            trialID += 1

    
        simID += 1
        objID += 1

    # save_master('mz',Total_array1)
    # save_master('contact_sum',Total_array2)
    print("ALL SAVED")
    
