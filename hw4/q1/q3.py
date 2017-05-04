import sys
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from scipy import misc
from scipy.misc import imsave


def plot(images, img_name):

    print(images.shape)
    images = images.reshape( images.shape[0], 64, 64)
    n = 10
    margin = 5
    img_width = img_height = 64
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched = np.zeros((width, height))
    for i in range(n):
        for j in range(n):
            img = images[i * n + j]
            stitched[(img_width + margin) * i: (img_width + margin) * i + img_width, \
                             (img_height + margin) * j: (img_height + margin) * j + img_height] = img
    # save the result to disk
#    imsave(img_name, stitched)
    scipy.toimage(stitched, cmin=0, cmax=255).save(img_name)

def load_data():
    x_test = []
    for i in range(10):
        for j in range(10):
            x_test.append( misc.imread('../faceExpressionDatabase/'+chr(ord('A')+i) \
                + str(j).rjust(2,'0') + '.bmp' ).flatten() )
    return np.array(x_test, 'float32')

def main():
    X = load_data()
#    plot(X, "faces.png")
    # Let the data matrix X be of n x p size,
    # where n is the number of samples and p is the number of variables
    n, p = X.shape
#    print( np.mean(X, axis=0).shape )
#    plt.imsave("average.png", arr=np.mean(X, axis=0).reshape(64,64), cmap=plt.get_cmap('gray'))

    # Center the data
    ori = np.copy(X)
    X_mean = np.mean(X, axis=0)
    X -= np.mean(X, axis=0)
    # we now perform singular value decomposition of X
    # "economy size" (or "thin") SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    S = np.diag(s)
    k = int(sys.argv[1])
    print(k)
    US_k = U[:, 0:k].dot(S[0:k, 0:k])
    Eigenvectors = V[:,:k].T
    Eigenvectors = preprocessing.normalize(Eigenvectors, norm='l2')

    Reconstruction = US_k.dot(Eigenvectors) + X_mean
    Reconstruction.reshape( Reconstruction.shape[0], 64, 64)
    Reconstruction = np.clip(Reconstruction, 0, 255)

    error = np.sqrt( ((ori.flatten() - Reconstruction.flatten())**2).mean())/256
    print(error)

main()
