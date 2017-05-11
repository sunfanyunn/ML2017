import sys
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from dataGenerator import data_generator
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

inputFile = sys.argv[1]
outputFile=sys.argv[2]
# 0.81 ==> good under 13
# 0.91 ==> great around 30
# 0.95 ==> great around 40
arr = [ 0.81, 0.85, 0.91, 0.95]
def predict(X, clf):
    xx = [ [ raw_predict(x, th) for th in arr ] for x in X ]
    return clf.predict( xx )

def writeResult(clf):
    X = np.load(inputFile)
    prd_dim = predict(X, clf)
    with open(outputFile, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['setId', 'LogDim'])
        for i in range(len(x_test)):
            csv_writer.writerow([i]+[prd_class[i]])

def raw_predict(X,threshold=0.95):

    pca = PCA()
    X_transformed = pca.fit_transform(X)
    n_samples = X.shape[0]

    # We center the data and compute the sample covariance matrix.
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
    eigenvalues = pca.explained_variance_

    Sum = sum(eigenvalues)
    ret = 0
    csum = 0
    for ind, x in enumerate(eigenvalues):
        if( csum > threshold*Sum ): return ind
        csum += x

def main():
#    generator = data_generator()
    data = np.load("data4_10000.npz")
    x_train, y_train = data['arr_0'], data['arr_1']
    clf = tree.DecisionTreeClassifier()
    print("fitting...")
    clf.fit(x_train, y_train)
    print("end fitting")

main()
