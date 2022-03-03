import load_data
import KNN
import cross_validation
import Minkowski

import pandas as pd
from functools import partial

data = load_data.load_data("MNIST_train_small.csv")

loss_df = pd.DataFrame(columns = list(range(1,21)))

for p in range(1,16): #test all p in range 1 up to and incl 15
    print(f"testing p = {p}")
    dist_met = partial(Minkowski.Minkowski_distance, p = p) #define a function that is the minkowski distance but with a set p
    loss_dict = cross_validation.loocv(data, distance_metric = dist_met)
    loss_df = loss_df.append(loss_dict, ignore_index=True)
    loss_df.to_csv(f"./up_to_p{p}.csv")
    
loss_df.index = list(range(1,16))
        