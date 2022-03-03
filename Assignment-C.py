import load_data
import KNN
import cross_validation
import Minkowski

import pandas as pd
from functools import partial

data = load_data.load_data("MNIST_train_small.csv")

loss_list = []

for p in range(1,16): #test all p in range 1 up to and incl 15
    print(f"testing p = {p}")
    dist_met = partial(Minkowski.Minkowski_distance, p = p) #define a function that is the minkowski distance but with a set p
    loss_dict = cross_validation.loocv(data, distance_metric = dist_met)
    loss_list.append(loss_dict)
    
    
#%% Convert loss_list to dataframe for easier interpretation

loss_df = pd.DataFrame(columns = list(range(1,21)))

for l in loss_list:
    loss_df.append(l)
    
loss_df.index = list(range(1,16))
        