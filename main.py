import load_data
import KNN
import time
from cross_validation import loocv
import json
import Minkowski
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt



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


def assignment_1b(path : str):
    train_data = load_data.load_data(path)
    err_dict = loocv(train_data)
    with open('result_b.json', 'w') as f:
        json.dump(err_dict, f)
    print("part b done.")


def plot_b(file_name):

    # plotting
    result_dict_b = {}
    with open(file_name) as result_b:
        result_file_b = json.load(result_b)
        result_str_b = json.dumps(result_file_b)
        result_dict_b = json.loads(result_str_b)

    x = list(result_dict_b.keys())
    y = list(result_dict_b.values())
    y_accuracy = []
    y_accuracy[:] = [1-value for value in y]
    # print(y_accuracy)

    plt.plot(x, y)
    plt.title("Errors acquired by LOOCV for each k value")
    plt.grid()
    plt.xlabel("k values")
    plt.ylabel("errors")
    plt.show()

    plt.plot(x, y_accuracy)
    plt.title("Accuracy values acquired by LOOCV for each k value")
    plt.grid()
    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.show()



def assignment_1e(path : str):
    """
    Executes research for assignment 1e
    :param path: path to the MNIST training dataset on your computer
    :return:
    """
    data = load_data.load_data(path,scale=True,percent=0.1)
    k_list = [3]#,3,5,10]
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

def assignment_1e2():
    '''
    Loads output files from assignment_1e and prints accuracy for each k value
    '''
    k_dict = {}
    for k in '1','3','5','10':
        f = open(f'k{k}.json')
        k_dict.update(json.load(f))
    print(k_dict)

def assignment_1f(k):
    '''
    returns the average loss on the test set for the knn
    :param k: the k value to be used for the knn
    '''
    test_data = load_data.load_data('MNIST_test.csv')
    train_data = load_data.load_data('MNIST_train.csv')

    model = KNN.KNN(train_data, k)
    print(model.calc_loss(test_data))


def assignment_1d(path : str):
    """
    Executes research for assignment 1e
    :param path: path to the MNIST training dataset on your computer
    :return:
    """
    p_list = [0.001,0.001,.99]#,3,5,10]
    k = 3
    p = 14
    dist_met = partial(Minkowski.Minkowski_distance, p=p)
    df_out = pd.DataFrame()
    train_dict = {}
    for perc in p_list:
        if perc == -10:
            data = load_data.load_data(path)
        else:
            data = load_data.load_data(path, scale=True,percent=perc)

        i = 0
        start = time.time()

        knn = KNN.KNN(data, k, distance_metric = dist_met)
        err = 0
        for point in data:
            result_y = knn.predict_point_LOOCV(point)
            if point[0] - result_y != 0:
                err += 1

            if i % 1000 == 0:
                print(f"K {k}, I {i}, time {int(time.time()-start)}, err {err}")
                #with open(f"result_e_k_{k}_i_{i}.txt", 'w') as f:
                #    f.write(str(err))
            i += 1
        # overall % in error
        tot_err = err / len(data)
        acc = 1-tot_err
        train_dict.update({str(perc)+str('acc'): acc})
        train_dict.update({str(perc)+str('time'): int(time.time()-start)})

    with open('result_e_partial.json', 'w') as f:
        json.dump(train_dict, f)
    print("program finished")

def assignment_1d_vis(path : str):
    df = pd.read_json('result_e_partial.json',typ='series')
    df = df.to_frame()
    df_time = df[df.index.str.contains('Time')]
    x = [0.1,0.2,0.5,0.8,0.9,1.1]
    y = df_time[0].to_list()
    
    plt.plot(x, y)
    plt.title("Time for different Percentages and Original Time")
    plt.grid()
    plt.xlabel("Percents (decimal)")
    plt.ylabel("Time (seconds)")
    plt.show()

    df_acc= df[df.index.str.contains('Acc')]
    y = df_acc[0].to_list()
    plt.plot(x, y)
    plt.title("Accuracy for different Percentages and Original Time")
    plt.grid()
    plt.xlabel("Percents (decimal)")
    plt.ylabel("Accuracy")
    plt.show()



if __name__ == "__main__":

    # assignment_1a()
    # assignment_1b("MNIST_train_small.csv")
    # assignment_1e("MNIST_train_small.csv")

    #assignment_1a()
    # assignment_1e2()
    # assignment_1f(3) #running time: 1h 7min 9sec. Loss: 0.0283
    #assignment_1d("MNIST_train_small.csv")
    #assignment_1d_vis(1)
    assignment_1e("MNIST_train.csv")
    #plot_b("result_b.json")
