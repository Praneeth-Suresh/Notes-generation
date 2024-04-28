# A word will be passed into the algorithm along with the segmentations; the algorithm will convert this word into a string of ASCII characters

import numpy as np
from hmmlearn.hmm import GaussianHMM

# The following is all the ASCII characters you may find
characters = """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}"""

image_width = 128
image_height = 128

def CleanImage():
    pass

def HMM():
    model = GaussianHMM(n_components=4, covariance_type="full", algorithm='viterbi')

    model.startprob_ = np.array([0.4, 0.3, 0.1, 0.2])
    model.transmat_ = np.array([[0.6, 0.2, 0.1, 0.1],
                                [0.3, 0.2, 0.2, 0.3],
                                [0.3, 0.3, 0.3, 0.1],
                                [0.4, 0.3, 0.1, 0.2]])
    # Since it is a Gaussian HMM
    model.means_ = np.array([[0.0, 0.0, 0.0], [3.0, -3.0, -1.5], [5.0, 10.0, 4.0], [1.0, 3.0, 6.0]])
    model.covars_ = np.tile(np.identity(3), (4, 1, 1))

    X, Z = model.sample(10)

    print("X (observed state) =",X,"Z (hidden state) =",Z)
    print(model.n_features)

    model.fit(X)

    Z2 = model.predict(X)

    print("Z2=", Z2)

HMM()