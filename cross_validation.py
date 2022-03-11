import numpy as num
from KNN import KNN
import time

def loocv(X, distance_metric=None):

    err_list = {}

    # test for k values from 1 to 20
    for k in range(1, 21):
        start = time.time()
        err = 0
        knn = KNN(X, k, distance_metric)

        # conduct loocv for each row
        for i in range(len(X)):

            # row to leave out
            test = X[i, :]
            # label of the row
            label = X[i, 0]

            # rest of the rows
            #train_X = num.concatenate(X[0:i, :], X[i+1:, :])

            #knn = KNN(train_X, k)
            # result from knn
            result_y = knn.predict_point_LOOCV(test)
            # error value for this point (might use a different calculation)
            if label - result_y != 0:
                err += 1

        # overall error
        err = err/len(X)
        err_list.update({k: err})
        print(f"finished k {k}, time: {int(time.time()-start)}")
    return err_list
