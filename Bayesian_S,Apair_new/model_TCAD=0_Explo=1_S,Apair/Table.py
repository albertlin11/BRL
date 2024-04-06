import pandas as pd
import csv
a = pd.read_csv("./RL_Input_parameter.csv")
b = a.values
table = "table_old_reward_prior=0.csv"

with open(table, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['network', 'batch size',
                     "best_timestep", "minimum", "count_before_best",
                     "total_count"])
for i in range(6):
    
    network = b[i][0]
    batch_size = b[i][1]

    

    explo = 0.0
    filepath = "./data_old_reward_prior=0_2/RL_network="+str(network)+ \
        "_batch_size="+ str(batch_size)+ ".csv"
    data = pd.read_csv(filepath)
    data_value = data.values

    best_timestep = data_value[50][1]


    minimum = data_value[50][3]

    count_before_best = data_value[51][1]

    total_count = data_value[51][3]

    with open(table, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([network, batch_size,
                         str(best_timestep), 
                         str(round(minimum,2)), 
                         str(count_before_best),
                         str(total_count)])