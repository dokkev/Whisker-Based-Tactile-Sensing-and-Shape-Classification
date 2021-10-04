import csv
import numpy as np
import os
import pathlib

def read_from_csv(dir):

    result = [] # array
    with open(dir,'r')as file:
        filecontent=csv.reader(file)
        result = list(filecontent)
    # print(result)

    return result


def read_from_csv_2(dir,val):

    dir = dir + str(val)+'.csv'

    result = [] # array
    with open(dir,'r')as file:
        filecontent=csv.reader(file)
        result = list(filecontent)
    # print(result)

    return result


def vector_plot_3d(x,y,z):
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z))

    return u,v,w


def save_data(dirname,data,type):
    dirout = '../results/'+str(type)
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.savetxt(str(dirout) + '/' +str(dirname), data, delimiter=',')

    # dirname = "concave20.obj_T" + format(trials, '03d') + "_N00"