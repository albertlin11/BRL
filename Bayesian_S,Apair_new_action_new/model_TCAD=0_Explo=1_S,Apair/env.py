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
    def new_list(self,number):
        if number == 1:
            return [0,1,2]
        if number == 2:
            return [1,2,3]
        if number == 3:
            return [2,3,4]
        if number == 4:
            return [3,4,5]
        
    def two_level(self,index,state):
        if index == 0:
            return state
        if index == 1:
            if state == 1:
                return 2
            else:
                return 1
        
    def new_action(self,state):
        action = []
        ion = state[0]
        dose = self.new_list(state[1])
        ion_energy = self.new_list(state[2])
        wavelength = state[3]
        rep = state[4]
        power = self.new_list(state[5])
        temp = state[6]
        for i in range(16):
            index = format(i,"04b")
            ion_para = self.two_level(int(index[0]),ion)
            wavelength_para = self.two_level(int(index[1]),wavelength)
            rep_para = self.two_level(int(index[2]),rep)
            temp_para = self.two_level(int(index[3]),temp)
            for j in range(len(dose)):
                dose_para = dose[j]
                for k in range(len(ion_energy)):
                    ion_energy_para = ion_energy[k]
                    for y in range(len(power)):
                        power_para = power[y]
                        action.append([ion_para,dose_para,ion_energy_para,
                                       wavelength_para,rep_para,power_para,temp_para])
        return action
    
    """
    It will find true value of each variable and get sheet resistance (Rs) from 
    dataset in response function. Finialy, it will return state, reward, done 
    (terminal), timestep, and Rs.
    """
    # Dataset = 0 use TCAD dataset
    # Dataset = 1 use experiment dataset
    def response(self, state, action):
        state = self.new_action(state)[action]
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
