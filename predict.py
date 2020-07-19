import numpy as np
import csv
import sys
import math
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def import_train_data(train_X_path,train_Y_path):
    train_X = np.genfromtxt(train_X_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_path, delimiter=',', dtype=np.float64)
    return train_X,train_Y

def compute_ln_norm_distance(vector1, vector2, n):
    Ln_norm = 0
    for i in range(len(vector1)):
        Ln_norm += pow(abs(vector1[i] - vector2[i]),n)
    Ln_norm = pow(Ln_norm,1/n)
    return Ln_norm

def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    
    index_dist = [ (i,compute_ln_norm_distance(train_X[i],test_example,n_in_ln_norm_distance)) for i in range(len(train_X))]
    index_dist.sort(key = lambda d: d[1])
    index = [ index_dist[i][0] for i in range(k)]
    return index
    
def mode(C):
    P = C[0]
    for i in range(1,len(C)):
        if C.count(C[i]) > C.count(P):
            P = C[i]
    return P

def classify_points_using_knn(train_X, train_Y, test_X, k, n_in_ln_norm_distance):
    classified_values = []
    for i in range(len(test_X)):
        KNNi = find_k_nearest_neighbors(train_X,test_X[i],k,n_in_ln_norm_distance)
        Ys = [train_Y[KNNi[j]] for j in range(len(KNNi))]
        classified_values.append(mode(Ys))

    return classified_values


def predict_target_values(test_X):
    train_X,train_Y = import_train_data("train_X_knn.csv","train_Y_knn.csv")
    K = 1
    predicted_values = classify_points_using_knn(train_X,train_Y,test_X,K,len(train_X[1]))

    return predicted_values
    


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 
