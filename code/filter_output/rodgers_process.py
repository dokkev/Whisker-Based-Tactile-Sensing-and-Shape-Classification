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
    objects_max = 6 # number of objects to process

    for objID in range(objects_max):
        objFile = objects[objID]

        trialID = 0    # inital number of trials to process
        trials_max = 1000 # final number of trials to process
        
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
            contact_sum = np.sum(contact_indicator,axis=0)

            contact_number = np.where(contact_sum>0, 1, 0)

            # img_contact_number = convert_contact_to_gray(contact_number)
            # save_image(dirname,img_contact_number,"rodgers_contact_number")

            moment = W.pick_protraction_moment(W.My,W.Mz)
 
            contact_number = W.add_convace_indicator_flat(contact_number,dirname)
            moment = W.add_convace_indicator_flat(moment,dirname)
            contact_sum = W.add_convace_indicator_flat(contact_sum,dirname)

            Total_array1.extend([moment])
            Total_array2.extend([contact_number])
            Total_array3.extend([contact_sum])


            print(dirname,"saved")
                
            trialID += 1

    
        simID += 1
        objID += 1

    save_master('moment',Total_array1)
    save_master('contact_number',Total_array2)
    save_master('contact_sum',Total_array3)
    print("ALL SAVED")
    
