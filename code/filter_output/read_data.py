import csv
import numpy as np
import os
import pathlib
import cv2
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat
import copy

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
    u = np.sin(np.pi * float(x)) * np.cos(np.pi * float(y)) * np.cos(np.pi * float(z))
    v = -np.cos(np.pi * float(x)) * np.sin(np.pi * float(y)) * np.cos(np.pi * float(z))
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * float(x)) * np.cos(np.pi * float(y)) * np.sin(np.pi * float(z)))

    return u,v,w


def save_data(dirname,data,type):
    dirout = '../results/'+str(type)
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.savetxt(str(dirout) + '/' +str(dirname) + '.csv', data, delimiter=',')
    

    # dirname = "concave20.obj_T" + format(trials, '03d') + "_N00"


def save_master(filename,data):
    dirout = '../results/'
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(str(dirout) + '/' +str(filename) + '.csv', data, delimiter=',')

def save_3d_array(filename,data):
    dirout = '../results/'
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)
    saveas = (str(dirout) + '/' +str(filename) + '.csv')
    savemat(saveas,data)

def append_data_to_list(list,data):

    if float(data) < 100:
        list.append(float(data))
    else:
        print("data seems too large! Data: ", data)
        pass

def convert_to_RGB(volume: np.ndarray, index: int):
    # Get absolute value
    volume = np.abs(volume)

    r = np.array(volume[:, :, 0])
    g = np.array(volume[:, :, 1])
    b = np.array(volume[:, :, 2])

    r = np.divide(r,np.max(r))
    g = np.divide(g,np.max(g))
    b = np.divide(b,np.max(b))

    rvolume = np.zeros((len(volume),len(volume[0]),3))
    rvolume[:,:,0] = r
    rvolume[:,:,1] = g
    rvolume[:,:,2] = b
    # print(np.shape(rvolume))

    rvolume = np.nan_to_num(rvolume)
    rvolume = np.multiply(rvolume,255)
    rvolume = rvolume.astype(np.uint8)
    # print(rvolume)
    img_arr = Image.fromarray(rvolume.astype('uint8'), 'RGB')

    return img_arr

# def convert_to_RGB(volume: np.ndarray, index: int):
#     scaler = MinMaxScaler(feature_range=(0, 255))
#     rvolume = scaler.fit_transform(volume)
#     img_arr = Image.fromarray(rvolume)


def convert_to_Gray(volume: np.ndarray):
    volume = np.abs(volume)
    rvolume = np.divide(volume,np.max(volume))
    rvolume = np.multiply(rvolume,255)
    rvolume = rvolume.astype(np.uint8)
    img_arr = Image.fromarray(rvolume.astype('uint8'))

    if img_arr.mode == "F" or img_arr.mode == "I":
        img_arr = img_arr.convert('L')
    return img_arr



def convert_contact_to_gray(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 1:
                data[i][j] = 255
            else:
                data[i][j] = 0
    
    img_arr = Image.fromarray(np.uint8(data),'L')


    return img_arr


def save_image(dirname,data,type):
    if 'concave' in dirname:
        dirout = '../results/images/' + str(type) + '/concave/'
    elif 'convex' in dirname:
        dirout = '../results/images/' + str(type) + '/convex/'
    else:
        dirout = '../results/images/' + str(type) + '/'
    directory = os.path.dirname(dirout)
    pathlib.Path(dirout).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_dir = dirout+str(dirname)+'.jpg'
    data.save(img_dir)

