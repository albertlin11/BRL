from prior_prob import prior_prob
import numpy as np
import numpy.ma as ma
from env import env
import os
import random
import random as python_random
import csv
import pandas as pd
from find_Rs import find_Rs
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("data_old_reward_radius=1_new_2", exist_ok = 1)


tf.random.set_seed(1)
os.environ['PYTHONHASHSEED'] = '0'
random.seed(1)
python_random.seed(1)
np.random.seed(1)

def decoder(action):
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

def change_TCAD(TCAD,reward,next_state_index,list):
    if next_state_index not in list:
        mul = abs(reward / TCAD[next_state_index])
        TCAD[next_state_index] = reward
        return TCAD, mul
    else:
        mul = 1
        return TCAD, mul

def save_prior_change_index(state,list):
    change_list = []
    i = 0
    while(i<1024):
        if i in list:
            i += 1
        else:
            test = np.abs(decoder(i)-state)
            if np.sum(test) == 1:
                change_list.append(i)
            i += 1
    return change_list

def prior_change(change_list,list,mul,TCAD):
    i = 0
    while(i < len(change_list)):
        if change_list[i] in list:
            i += 1
        else:
            TCAD[change_list[i]] = round(TCAD[change_list[i]] * mul,2)
            i += 1
    return TCAD

def state_to_index(state):
    binary = [0,0,0,0,0,0,0,0,0,0]
    binary[3] = str(state[0] // 2)
    binary[0] = str((state[1] - 1) // 2)
    binary[4] = str((state[1] - 2 * int(binary[0])) // 2)
    binary[1] = str((state[2] - 1) // 2)
    binary[5] = str((state[2] - 2 * int(binary[1])) // 2)
    binary[6] = str(state[3] // 2)
    binary[7] = str(state[4] // 2)
    binary[2] = str((state[5] - 1) // 2)
    binary[8] = str((state[5] - 2 * int(binary[2])) // 2)
    binary[9] = str(state[6] // 2)
    out = ""
    for i in range(10):
        out = out + binary[i]
    return int(out,2)+1


class MakeModel:
    def __init__(self,network = 0, batch_size=2,filepath="Save_Model.h5"):
        self.network = network
        self.model = self.create_model()
        self.batch_size = batch_size
        self.filepath = filepath

    def create_model(self):
        if self.network == 2:
            input_layer = Input(shape = (2,), name = 'input_layer')
            x = Dense(100,activation="tanh")(input_layer)
            x = Dense(100,activation="tanh")(x)
            x = Dense(28,activation="tanh")(x)
            output_layer = Dense(1, name = "output_layer")(x)
            model = Model(inputs = input_layer, outputs = output_layer)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mse')
            return model
        else:
            input_layer = Input(shape = (2,), name = 'input_layer')
            x = Dense(100,activation="tanh")(input_layer)
            x = Dense(100,activation="tanh")(x)
            x = Dense(100,activation="tanh")(x)
            x = Dense(100,activation="tanh")(x)
            x = Dense(100,activation="tanh")(x)
            x = Dense(28,activation="tanh")(x)
            output_layer = Dense(1, name = "output_layer")(x)
            model = Model(inputs = input_layer, outputs = output_layer)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mse')
            return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def train(self, state, target):
        callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=self.filepath,
        save_weights_only=True,
        monitor='loss',
        save_best_only=True)
        self.model.fit(state, target, epochs=1000, batch_size=self.batch_size, 
        callbacks=[callback], verbose=0)
    
    def load_model(self):
        self.model.load_weights(self.filepath)

    def evaluation(self, state, target):
        self.model.evaluate(state,target,batch_size=self.batch_size)

def new_state(number):
        if number == 1:
            return [0,1,2]
        if number == 2:
            return [1,2,3]
        if number == 3:
            return [2,3,4]
        if number == 4:
            return [3,4,5]

def two_level(index,state):
    if index == 0:
        return state
    if index == 1:
        if state == 1:
            return 2
        else:
            return 1

def new_action(state):
        action = []
        mask = np.zeros(432)
        mask_can_select = np.zeros(432)
        ion = state[0]
        dose = new_state(state[1])
        ion_energy = new_state(state[2])
        wavelength = state[3]
        rep = state[4]
        power = new_state(state[5])
        temp = state[6]
        mask_index = 0
        for i in range(16):
            index = format(i,"04b")
            ion_para = two_level(int(index[0]),ion)
            wavelength_para = two_level(int(index[1]),wavelength)
            rep_para = two_level(int(index[2]),rep)
            temp_para = two_level(int(index[3]),temp)
            for j in range(len(dose)):
                dose_para = dose[j]
                for k in range(len(ion_energy)):
                    ion_energy_para = ion_energy[k]
                    for y in range(len(power)):
                        power_para = power[y]
                        parameter = [ion_para,dose_para,ion_energy_para,
                                     wavelength_para,rep_para,power_para,temp_para]
                        if 0 in parameter:
                            mask[mask_index] = 1
                            save_index = 1
                        elif 5 in parameter:
                            mask[mask_index] = 1
                            save_index = 1
                        else:
                            save_index = state_to_index(parameter)
                            mask_can_select[mask_index] = 1
                        save_index = save_index -1
                        mask_index += 1
                        action.append(save_index)
        mask_can_select = mask_can_select / np.sum(mask_can_select)
        return action,mask,mask_can_select

def generate_all_input(state):
    total = []
    state = np.array(state)
    for i in range(1,433):
        new = np.append(state,i)
        total.append(list(new))
    total = np.reshape(total,(-1,2))
    return total

def state_prior(prior,action_index,mask):
    new_prior = np.array([])
    for i in range(len(action_index)):
        if mask[i] == 0:
            new_prior = np.append(new_prior,prior[action_index[i]])
        else:
            new_prior = np.append(new_prior,-1000)
    return new_prior


a = pd.read_csv("RL_Input_parameter.csv")
a = a.values

for k in range(6):

    tf.random.set_seed(1)
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(1)
    python_random.seed(1)
    np.random.seed(1)

    network_num = a[k][0]
    batch_size_num = a[k][1]
    radius = a[k][2]
    
    filepath2 = "./data_old_reward_radius=1_new_2/RL_network="+str(network_num)+ "_batch_size="+\
        str(batch_size_num)+"_radius=1.csv"

    with open(filepath2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ion', 'Dose', 'Ion Energy', 'Wavelength',
                            'Rep. Rate', 'Power', 'Temperature', 'Rs'])

    
    input_list = np.array([[1,1],[1025,432]])
    input_list = np.log10(input_list)
    in_standard = MinMaxScaler()
    input_list = in_standard.fit_transform(input_list)


    out_standard = MinMaxScaler()
    out_list = np.array([0,-3]).reshape(-1,1)
    out_list = out_standard.fit_transform(out_list)


    environment = env(reward_func=0)
    prior = prior_prob(reward=0)
    prior = np.array(prior)

    model1 = MakeModel(network=int(network_num),batch_size=int(batch_size_num))
    model2 = MakeModel(network=int(network_num),batch_size=int(batch_size_num),filepath="SaveModel2.h5")
    

    save_state = []
    save_reward = []
    save = []
    best = 1e11
    besttimestep = 0
    epsilon = 0.5
    difference=[]

    for ep in range(5):
        state = environment.reset()
        timestep = 0
        done = False
        while not done:
            action_index, mask, mask_can_select = new_action(state)
            new_prior = state_prior(prior,action_index,mask)
            state_index = state_to_index(state)
            timestep+=1

            if ep != 0 and timestep == 1:
                epsilon = round(epsilon * 0.9,2)

            if ep==0:
                divide = 1
            else:
                divide = ((ep-1)*10+timestep)**2

            if timestep % 5 == 1:
                select = np.random.choice(2)

            if np.random.randint(1,11)/10 <= epsilon:
                action = np.random.choice(432,p=mask_can_select)
            else:
                pred_input = generate_all_input(state_index)
                pred_input = np.log10(pred_input)
                pred_input = in_standard.transform(pred_input)
                if select == 1:
                    pred = model1.predict(pred_input)
                    pred = out_standard.inverse_transform(pred)
                    pred = np.reshape(pred,(1,-1))[0]
                    pred = -(10 ** (-pred))
                else:
                    pred = model2.predict(pred_input)
                    pred = out_standard.inverse_transform(pred)
                    pred = np.reshape(pred,(1,-1))[0]
                    pred = -(10 ** (-pred))
                pred = pred + 2*new_prior / divide
                new_pred = ma.array(pred,mask = mask)
                action = np.argmax(new_pred)
                

            next_state,reward,done,env_timestep,Rs,true_label = environment.response(state,action)
            print("timestep",(ep*10+timestep),"Rs",Rs)

            true_param = np.append(true_label, Rs)
            
            with open(filepath2, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(true_param)
            
            next_state_index = state_to_index(next_state)
            memory = [state_index,action+1,next_state_index,done,next_state]

            if Rs >= 1000:
                new_Rs = 1000
            else:
                new_Rs = Rs


            training_in = []
            training_out1 = []
            training_out2 = []

            if memory not in save_state:
                save_state.append(memory)
                save_reward.append(new_Rs)

            for j in range(len(save_state)):
                action_index, mask, mask_can_select = new_action(save_state[j][4])
                new_prior = state_prior(prior,action_index,mask)
                if save_state[j][3] == 1:
                    training_in.append(save_state[j][0:2])
                    save_q = -np.log10(save_reward[j])
                    training_out1.append(save_q)
                    training_out2.append(save_q)
                else:
                    training_in.append(save_state[j][0:2])
                    pred_input = generate_all_input(save_state[j][2])
                    pred_input = np.log10(pred_input)
                    pred_input = in_standard.transform(pred_input)
                    pred1 = model1.predict(pred_input)
                    pred2 = model2.predict(pred_input)
                    pred1 = out_standard.inverse_transform(pred1)
                    pred2 = out_standard.inverse_transform(pred2)
                    pred1 = np.reshape(pred1,(1,-1))[0]
                    pred2 = np.reshape(pred2,(1,-1))[0]
                    pred1 = -(10 ** (-pred1))
                    pred2 = -(10 ** (-pred2))
                    pred1 = pred1 + 2*new_prior/divide
                    pred2 = pred2 + 2*new_prior/divide
                    new_pred1 = ma.array(pred1,mask = mask)
                    new_pred2 = ma.array(pred2,mask = mask)
                    action1 = np.argmax(new_pred1)
                    action2 = np.argmax(new_pred2)
                    save_q1 = -np.log10(save_reward[j]+0.2*np.abs(pred1[action1]))
                    save_q2 = -np.log10(save_reward[j]+0.2*np.abs(pred2[action2]))
                    if save_q1 <= -3:
                        save_q1 = -3
                    if save_q2 <= -3:
                        save_q2 = -3    
                    training_out1.append(save_q1)
                    training_out2.append(save_q2)
            train_in = np.log10(training_in)
            #train_in = training_in
            train_in = in_standard.transform(np.reshape(train_in,(-1,2)))
            train_out1 = out_standard.transform(np.reshape(training_out1,(-1,1)))
            train_out2 = out_standard.transform(np.reshape(training_out2,(-1,1)))

            if Rs < best:
                best = Rs
                besttimestep = (ep*10+timestep)
                
            if (ep*10+timestep) <= 49:
                model1.train(train_in,train_out1)
                model2.train(train_in,train_out2)
            
            model1.load_model()
            model2.load_model()
            
            next_state_index = state_to_index(next_state)
            next_state_index = next_state_index - 1

            prior, mul= change_TCAD(prior,reward,next_state_index,save)
            if radius == 1:
                change_list = save_prior_change_index(next_state,save)
                prior = prior_change(change_list,save,mul,prior)
            
            state = next_state
            
            if next_state_index not in save:
                save.append(next_state_index)
    print("Best Timestep:", besttimestep, 
              "Minimum Rs: ", best)



    if filepath2 != 0:
        with open(filepath2, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Best Timestep", besttimestep, 
                                "Minimum Rs", best])
        count, count_before_best = find_Rs(filepath2)
        with open(filepath2, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Count berfore best timestep", 
                                count_before_best, 
                                "Total count", count])
