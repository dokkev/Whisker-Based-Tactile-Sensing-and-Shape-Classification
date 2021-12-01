import numpy as np

def get_Rx(theta):
    R = np.array([[1.,0.,0.],[0., np.cos(theta), -np.sin(theta)],[0.,np.sin(theta), np.cos(theta)]])
    return R

def get_Ry(theta):
    R = np.array([[np.cos(theta),0.,np.sin(theta)],[0., 1., 0.],[-np.sin(theta),0., np.cos(theta)]])
    return R

def get_Rz(theta):
    theta = theta
    R = np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta), np.cos(theta), 0.],[0.,0., 1.]])
    return R

def update_roll(state,turn_size,next_step,orientation):
    state[5] += turn_size

    if np.abs(state[5])>=2*np.pi:
        state[5] = 0.
    

    # get roll
    Rx = get_Rx(state[5])

    next_step = np.dot(Rx,orientation)
    # next_step = np.dot(Rx,next_step)
    return next_step

def update_yaw(state,turn_size,next_step,orientation):
    
    state[3] += turn_size

    if np.abs(state[3])>=2*np.pi:
        state[3] = 0.

    # get yaw
    Rz = get_Rz(state[3])

    next_step = np.dot(Rz,orientation)

    return next_step