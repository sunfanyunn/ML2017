
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from dataGenerator import data_generator
import matplotlib.pyplot as plt
from sklearn import tree



def raw_predict(X,threshold=0.95):

    pca = PCA()
    X_transformed = pca.fit_transform(X)
    n_samples = X.shape[0]

    # We center the data and compute the sample covariance matrix.
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
    eigenvalues = pca.explained_variance_
    return eigenvalues

def main():
    generator = data_generator()
    dim = [np.random.randint(1, 61) for _ in range(100)]
    X = []
    for i,d in enumerate(dim):
        data = generator.generate(d, np.random.randint(10000,100001))
        X.append( raw_predict(data, 0) )
        print(i)
    np.savez("valid100.npz", X, dim)

main()
