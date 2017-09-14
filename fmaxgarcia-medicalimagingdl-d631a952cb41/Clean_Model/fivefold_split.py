import numpy as np
from load_data import load_data
import os
import pickle
import random

import sys

# control_dir = "../../normal_70_74_20/"
# disease_dir = "../../disease_70_79_20/"
control_dir = sys.argv[1]
disease_dir = sys.argv[2]

X0, Y0, X1, Y1 = load_data(control_dir, disease_dir)

X = np.vstack( (X0, X1) )
Y = np.vstack( (Y0, Y1) )

indices = range(X.shape[0])
random.shuffle(indices)

length = X.shape[0] // 5
print(length)
print(X.shape)
for fold in range(5):
    directory = "./fold_" + str(fold)
    print(directory)
    if os.path.isdir(directory) == False:
        os.mkdir(directory)

    if fold == 0:
        Xtrain = X[indices[length:]]
        Ytrain = Y[indices[length:]]
        Xtest = X[indices[:length]]
        Ytest = Y[indices[:length]]
    elif fold == 4:
        Xtrain = X[indices[:length*4]]
        Ytrain = Y[indices[:length*4]]
        Xtest = X[indices[length*4:]]
        Ytest = Y[indices[length*4:]]
    else:
        print(fold*length)
        print((fold+1)*length)
        Xtrain = np.vstack( (X[indices[:fold*length]], X[indices[(fold+1)*length]:]) )
        Ytrain = np.vstack( (Y[indices[:fold*length]], Y[indices[(fold+1)*length]:]) )
        Xtest = X[indices[fold*length:(fold+1)*length]]
        Ytest = Y[indices[fold*length:(fold+1)*length]]

    pickle.dump( (Xtrain, Ytrain), open(directory+"/train.pkl", "wb"))
    pickle.dump( (Xtest, Ytest), open(directory+"/test.pkl", "wb"))



