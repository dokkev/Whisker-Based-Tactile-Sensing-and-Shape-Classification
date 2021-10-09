#!/usr/bin/env python

from read_data import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


def vector_plot(dynamic_data,bool):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # define xyz
    x,y,z = dynamic_data[0],dynamic_data[1],dynamic_data[2]

    # define vectors
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
        np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=bool)
    plt.show()

def multiplot(dynamic_data,title):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(title)
    x = np.linspace(0, 1, len(dynamic_data[0]))
    ax1.plot(x, dynamic_data[0])
    ax2.plot(x, dynamic_data[1])
    ax3.plot(x, dynamic_data[2])
    ax1.set_title("_x")
    ax2.set_title("_y")
    ax3.set_title("_z")
    plt.show()




if __name__ == '__main__':
    
    # read data
    F_concave = genfromtxt('../results/F_concave_total.csv',delimiter=',')
    # M_concave = genfromtxt('../results/M_concave_total.csv',delimiter=',')
    F_concave20_T000_N00 = genfromtxt('../results/f/concave20.obj_T000_N00.csv',delimiter=',')
    M_concave20_T000_N00 = genfromtxt('../results/m/concave20.obj_T000_N00.csv',delimiter=',')
    F_convex20_T000_N00 = genfromtxt('../results/f/convex20.obj_T000_N11.csv',delimiter=',')
    M_convex20_T000_N00 = genfromtxt('../results/m/convex20.obj_T000_N11.csv',delimiter=',')
    # F_convex = genfromtxt('../results/F_convex_total.csv',delimiter=',')
    # F_convex = read_from_csv_2('../results/','F_convex_total')
    print("data loaded")
    # prin


    # transpose
    F_concave = np.transpose(F_concave)
    # F_convex = np.transpose(F_convex)
    # print(F_concave)
    F_concave20_T000_N00 = np.transpose(F_concave20_T000_N00)
    M_concave20_T000_N00 = np.transpose(M_concave20_T000_N00)
    F_convex20_T000_N00 = np.transpose(F_convex20_T000_N00)
    M_convex20_T000_N00 = np.transpose(M_convex20_T000_N00)

    # print(F_concave20_T000_N00)

    # vector_plot(F_concave,False)
    multiplot(F_concave,'F_concave')
    # multiplot(F_concave20_T000_N00,'F_concave')
    # multiplot(M_concave20_T000_N00,'M_concave')
    # multiplot(F_convex20_T000_N00,'F_convex')
    # multiplot(M_convex20_T000_N00,'M_convex')





