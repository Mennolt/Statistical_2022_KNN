import load_data
import KNN

if __name__ == "__main__":
    data = load_data.load_data("MNIST_train_small.csv")
    print(data)
    test_data = data[0][1:]
    classifier = KNN.KNN(data, 10)
    classifier.predict_point(test_data)