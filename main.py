import load_data
import KNN
import time
import cross_validation
import json
import Minkowski
from functools import partial



def assignment_1a():
    # start = time.time()
    data = load_data.load_data("MNIST_train_small.csv")
    test_data = load_data.load_data("MNIST_test_small.csv")
    # print(f"data loading: {time.time()-start}")
    # #print(data)
    # test_data = data[0][1:]
    # classifier = KNN.KNN(data, 10)
    # print(classifier.predict_point(test_data))
    # print(f"total time: {time.time()-start}")
    #
    # test_LOOCV = data[1]
    # print(classifier.predict_point_LOOCV(test_LOOCV))

    # first experiment
    # start = time.time()
    # test_err_list = cross_validation.loocv(test_data)
    # print(f"test done {time.time() - start}")
    # train_err_list = cross_validation.loocv(data)
    # print(f"train done {time.time()-start}")

    # now using the actual things we need...
    train_dict = {}
    test_dict = {}


    for k in range(1, 21):
        start = time.time()

        knn = KNN.KNN(data, k)
        err = 0
        for point in data:
            result_y = knn.predict_point_LOOCV(point)
            if point[0] - result_y != 0:
                err += 1
        # overall % in error
        end_err = err / len(data)
        acc = 1 - end_err
        #print(end_err, err, len(data))
        train_dict.update({k: acc})

        err = 0
        for point in test_data:
            result_y = knn.predict_point(point[1:])
            if point[0] - result_y != 0:
                err += 1
        # overall % in error
        end_err = err / len(data)
        acc = 1 - end_err
        test_dict.update({k: acc})

    print(train_dict)
    print(test_dict)

    result_data = {"train": train_dict, "test": test_dict}
    #with open('result_a.json', 'w') as f:
    #    json.dump(result_data, f)


def assignment_1e(path : str):
    """
    Executes research for assignment 1e
    :param path: path to the MNIST training dataset on your computer
    :return:
    """
    data = load_data.load_data(path)
    k_list = [1]#,3,5,10]
    p = 14
    dist_met = partial(Minkowski.Minkowski_distance, p=p)

    train_dict = {}
    for k in k_list:
        i = 0
        start = time.time()

        knn = KNN.KNN(data, k, distance_metric = dist_met)
        err = 0
        for point in data:
            result_y = knn.predict_point_LOOCV(point)
            if point[0] - result_y != 0:
                err += 1

            if i % 100 == 0:
                print(f"K {k}, I {i}, time {int(time.time()-start)}")
            i += 1
        # overall % in error
        tot_err = err / len(data)
        acc = 1-tot_err
        train_dict.update({k: acc})

    with open('result_e_partial.json', 'w') as f:
        json.dump(train_dict, f)
    print("program finished")



if __name__ == "__main__":
    #assignment_1a()
    assignment_1e("MNIST_train.csv")