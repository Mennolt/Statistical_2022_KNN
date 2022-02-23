import load_data
import KNN
import time

if __name__ == "__main__":
    start = time.time()
    data = load_data.load_data("MNIST_train_small.csv")
    print(f"data loading: {time.time()-start}")
    #print(data)
    test_data = data[0][1:]
    classifier = KNN.KNN(data, 10)
    print(classifier.predict_point(test_data))
    print(f"total time: {time.time()-start}")

    test_LOOCV = data[1]
    print(classifier.predict_point_LOOCV(test_LOOCV))