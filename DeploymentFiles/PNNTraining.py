# PNN stands for a Probabilistically Weighted Neural Network.
# This file deals with training the PNN; both the NN segment and the Markov Model that gives the probabilitic weightings.

# If in doubt, refer to 20th August entry in the Notebook.

"""
Training NN:
To train the neural network use the image database in this folder. 
You need to include special characters as well because the NN needs to be able to recorgnise everything.
If the database does not have special characters, make a special arrangement for trying to train these.

Neural network structure: CNN
Take this for reference: https://keras.io/examples/vision/handwriting_recognition/
If the above is insufficient, make chages to expand or otherwise perfect the Neural Network.
"""

import numpy as np
import tensorflow as tf
import pandas as pd

# Pre-processing
## Features to implement: 
# - image multiplication (make more samples 1:20 ratio)
# - standardise image sizing (to 128 by 128) 
# - standardise pixel value limits (0 to 1)

src = pd.read_csv()