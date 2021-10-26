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
    Rz = get_Rz(state[5])
    # get yaw
    Rx = get_Rx(state[4])

    next_step = np.dot(Rz,orientation)
    # next_step = np.dot(Rx,next_step)
    return next_step

def update_pitch(state,turn_size,next_step,orientation):
    
    state[4] += turn_size

    if np.abs(state[4])>=2*np.pi:
        state[4] = 0.
    
    # get roll
    Rz = get_Rz(state[5])
    # get yaw
    Ry = get_Ry(state[4])

    next_step = np.dot(Ry,orientation)

    return next_step