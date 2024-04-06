import pandas as pd

def find_Rs(filepath):
    a1 = pd.read_csv(filepath, encoding = 'unicode_escape')
    b1 = a1.values
    best_timestep = b1[50,1]
    path = []
    count = 0
    count_before_best = 0
    
    for i in range(0,50):
        if i == 0:
            path.append(b1[i])
            count += 1
            count_before_best += 1
        else:
            j = 0
            same = 0
            while(j < len(path) and same == 0):
                k = 0
                same_number = 0
                while(k < 7):
                    if b1[i][k] == path[j][k]:
                        same_number += 1
                    k += 1
                if same_number == 7:
                    same = 1
                j += 1
            if same == 0:
                count += 1
                path.append(b1[i])
            if i == best_timestep - 1:
                count_before_best = count
    return count, count_before_best