import pandas as pd
import numpy as np
import os
from scipy.interpolate import griddata
import random
import random as python_random

os.environ['PYTHONHASHSEED'] = '0'
python_random.seed(1)
np.random.seed(1)

### Import Dataset ###
a1 = pd.read_csv('./dataset_new.csv', encoding = 'unicode_escape')
b1 = a1.values
b1[:,1] = b1[:,1]/1e14

### Date Processing ###
for i in range(len(b1)):
    if b1[i,6] < 35 and b1[i,6] > 15:
        b1[i,6] = 25
    if b1[i,6] > 290 and b1[i,6] < 310:
        b1[i,6] = 300
    if b1[i,3] == 532:
        if b1[i,5] > 1.4 and b1[i,5] < 1.6:
            b1[i,5] = 1.5
        if b1[i,5] > 2.9 and b1[i,5] < 3.1:
            b1[i,5] = 3
        if b1[i,5] > 4.4 and b1[i,5] < 4.6:
            b1[i,5] = 4.5
        if b1[i,5] > 5.9 and b1[i,5] < 6.1:
            b1[i,5] = 6
    if b1[i,3] == 355:
        if b1[i,4] == 50:
            if b1[i,5] > 0.9 and b1[i,5] < 1.1:
                b1[i,5] = 1
            if b1[i,5] > 1.56 and b1[i,5] < 1.76:
                b1[i,5] = 1.66
            if b1[i,5] > 2.23 and b1[i,5] < 2.43:
                b1[i,5] = 2.33
            if b1[i,5] > 2.9 and b1[i,5] < 3.1:
                b1[i,5] = 3
        if b1[i,4] == 100:
            if b1[i,5] > 0.4 and b1[i,5] < 0.6:
                b1[i,5] = 0.5
            if b1[i,5] > 0.9 and b1[i,5] < 1.1:
                b1[i,5] = 1
            if b1[i,5] > 1.4 and b1[i,5] < 1.6:
                b1[i,5] = 1.5
            if b1[i,5] > 1.9 and b1[i,5] < 2.1:
                b1[i,5] = 2
x_exp = b1[:, 0:7]
y_exp = b1[:, 16]

### Create environment ###
class env:
    def __init__(self, state = 7, action = 1024, 
                 reward_func = 0, max_timestep = 10):
        self.state = state
        self.action = action
        self.reward_func = reward_func
        self.max_timestep = max_timestep
        self.timestep = 0
        self.minimum = 1e11
    
    def get_state(self):
        return self.state
    
    def get_action(self):
        return self.action
    
    """
    Reset to initial state and make timestep be zero.
    """
    def reset(self):
        state = np.array([1, 3, 3, 2, 2, 1, 1])
        self.timestep = 0
        return state
        
    def reward(self, value):
        return  -value
    
    def reward_log(self, value):
        return -((np.log10(value)) ** 2)
    
    ### Enconder will transform action to a specific state ###
    """ 
    It will transform action to a binary number and first three number will 
    specify classes of dose, ion energy, and power, and the other numbers are
    state of each variable.
    """    
    def decoder(self,action):
        binary = format(action, '010b')
        choice = binary[:3]

        ion = int(binary[3]) + 1
        dose = 2 * int(choice[0]) + int(binary[4]) + 1
        ion_energy = 2 * int(choice[1]) + int(binary[5]) + 1
        wavelength = int(binary[6]) + 1
        rep_rate = int(binary[7]) + 1
        power = 2 * int(choice[2]) + int(binary[8]) + 1
        temp = int(binary[9]) + 1
        new_state = np.array([ion, dose, ion_energy, wavelength, 
                              rep_rate, power, temp])
        return new_state
    
    """
    It will find true value of each variable and get sheet resistance (Rs) from 
    dataset in response function. Finialy, it will return state, reward, done 
    (terminal), timestep, and Rs.
    """
    # Dataset = 0 use TCAD dataset
    # Dataset = 1 use experiment dataset
    def response(self, action):
        state = self.decoder(action)
        x = state
        y = np.array([0,0,0,0,0,0,0], dtype=float)
        if x[0] == 1:
            y[0] = 1
        if x[0] == 2:
            y[0] = 2

        if x[1] == 1:
            y[1] = 5
        elif x[1] == 2:
            y[1] = 8
        elif x[1] == 3:
            y[1] = 20
        elif x[1] == 4:
            y[1] = 50

        if x[2] == 1:
            y[2] = 10
        elif x[2] == 2:
            y[2] = 25
        elif x[2] == 3:
            y[2] = 40
        elif x[2] == 4:
            y[2] = 55

        if x[3] == 1:
            y[3] = 355
        if x[3] == 2:
            y[3] = 532

        if x[4] == 1:
            y[4] = 50
        if x[4] == 2:
            y[4] = 100

        if y[3] == 532:
            if x[5] == 1:
                y[5] = 1.5
            elif x[5] == 2:
                y[5] = 3
            elif x[5] == 3:
                y[5] = 4.5
            elif x[5] == 4:
                y[5] = 6
        if y[3] == 355:
            if y[4] == 50:
                if x[5] == 1:
                    y[5] = 1
                elif x[5] == 2:
                    y[5] = 1.66
                elif x[5] == 3:
                    y[5] = 2.33
                elif x[5] == 4:
                    y[5] = 3
            if y[4] == 100:
                if x[5] == 1:
                    y[5] = 0.5
                elif x[5] == 2:
                    y[5] = 1
                elif x[5] == 3:
                    y[5] = 1.5
                elif x[5] == 4:
                    y[5] = 2
        if x[6] == 1:
            y[6] = 25
        if x[6] == 2:
            y[6] = 300

        
        Rs = griddata(x_exp, y_exp, y, method ='nearest')
        Rs = Rs[0]
        
        if self.reward_func == 0:
            reward = self.reward(Rs)
        else:
            reward = self.reward_log(Rs)
            
        self.timestep += 1
        if self.timestep >= self.max_timestep:
            done  = True
        else:
            done = False

        return state, reward, done, self.timestep, Rs, y
