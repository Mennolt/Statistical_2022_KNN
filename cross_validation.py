import numpy as num
from KNN import KNN

def loocv(X):

    err_list = {}
    # test for k values from 1 to 20
    for k in range(1, 21):
        err = 0

        # conduct loocv for each row
        for i in range(len(X)):
            # row to leave out
            test = X[i, :]
            # rest of the rows
            train_X = num.concatenate(X[0:i, :], X[i+1:, :])

            knn = KNN(train_X, k)
            # result from knn
            result_y = knn.predict_point(test)
            # error value for this point (might use a different calculation)
            err += (X[i, 0]-result_y)**2

        # overall error
        err = err/len(X)
        err_list.update({k: err})

    return err_list
