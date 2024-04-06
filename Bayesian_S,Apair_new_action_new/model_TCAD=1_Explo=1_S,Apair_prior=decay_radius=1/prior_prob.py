import numpy as np
import pandas as pd
from scipy.interpolate import griddata
a2 = pd.read_csv('./TCAD.csv', encoding = 'unicode_escape')
b2 = a2.values
b2[:,1] = b2[:,1]/1e14

for i in range(len(b2)):
    if b2[i,6] < 35 and b2[i,6] > 15:
        b2[i,6] = 25
    if b2[i,6] > 290 and b2[i,6] < 310:
        b2[i,6] = 300
    if b2[i,3] == 532:
        if b2[i,5] > 1.4 and b2[i,5] < 1.6:
            b2[i,5] = 1.5
        if b2[i,5] > 2.9 and b2[i,5] < 3.1:
            b2[i,5] = 3
        if b2[i,5] > 4.4 and b2[i,5] < 4.6:
            b2[i,5] = 4.5
        if b2[i,5] > 5.9 and b2[i,5] < 6.1:
            b2[i,5] = 6
    if b2[i,3] == 355:
        if b2[i,4] == 50:
            if b2[i,5] > 0.9 and b2[i,5] < 1.1:
                b2[i,5] = 1
            if b2[i,5] > 1.56 and b2[i,5] < 1.76:
                b2[i,5] = 1.66
            if b2[i,5] > 2.23 and b2[i,5] < 2.43:
                b2[i,5] = 2.33
            if b2[i,5] > 2.9 and b2[i,5] < 3.1:
                b2[i,5] = 3
        if b2[i,4] == 100:
            if b2[i,5] > 0.4 and b2[i,5] < 0.6:
                b2[i,5] = 0.5
            if b2[i,5] > 0.9 and b2[i,5] < 1.1:
                b2[i,5] = 1
            if b2[i,5] > 1.4 and b2[i,5] < 1.6:
                b2[i,5] = 1.5
            if b2[i,5] > 1.9 and b2[i,5] < 2.1:
                b2[i,5] = 2
x_tcad = b2[:, 0:7]
y_tcad = b2[:, 17]



def decoder(action):
        binary = format(action-1, '010b')
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

def prior_prob(reward = 0):
    save_data = []
    for i in range(1,1025):
        state = decoder(i)
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

        Rs = griddata(x_tcad, y_tcad, y, method ='nearest')
        save_data.append(Rs[0])
    if reward == 0:
        prior_prob = list(map(lambda x: -x, save_data))
    else:
        prior_prob = list(map(lambda x: -(np.log10(x)) ** 2, save_data))
    return prior_prob