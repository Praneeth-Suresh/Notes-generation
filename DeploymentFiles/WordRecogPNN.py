# PNN stands for a Probabilistically Weighted Neural Network.
# This file uses the trained PNN from PNNTraining.py to classify and segment images at the same time.

# The input is going to be: previous character and image of current character
# If there is no previous character, input "-"; index = 100

import os
import pickle
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models

from PreProcessImage import preprocess_image, CharacterSegmentation, pad_image


root = """data\\"""
img_size = (128, 128)
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
       'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
       'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
       'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
learning_rate = 0.5
base_path = """C://Work//Programming//Hackathons//Reprose//Notes generation//DeploymentFiles//data//"""
words_list = []
max_length = 10 # This is to limit the code so that it can be run locally

# We get the file names from the datatset with which we can open the images later

count = 0
words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    
    words_list.append(line.strip())

    count += 1
    if count == max_length:
        break

print(len(words_list))

np.random.shuffle(words_list)

print(f"Total samples: {len(words_list)}")

# Build the data input pipeline

base_image_path = os.path.join(base_path, "words")

def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples

test_img_paths, test_labels = get_image_paths_and_labels(words_list)

def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels

test_labels_cleaned = clean_labels(test_labels)

AUTOTUNE = tf.data.AUTOTUNE

def process_images_labels(image_path, label):
    print(image_path)
    image, rotate, shear = preprocess_image(image_path)
    print("Pre-processing metrics generated")
    # this can produce errors; need to add a try catch here
    return CharacterSegmentation(label, image_path, rotate, shear)

MisSegmented = 0


# We build the PNN model

model = models.Sequential()

# Here's the convnet
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# This is the dense net that parses the output from the convnet
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(len(chars)))

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt)

""""In the time that the PNN is being developed, the checkpoints will remain commented.
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(os.getcwd() + "/Checkpoints/" + "PNN").expect_partial()
"""

def GetProbs(prev): # The input is the index of the character, not the character itself

    df2 = pd.read_csv('data//TransitionModel.csv', header=None)

    sum = df2.loc[prev].sum()

    return df2.loc[prev].tolist() / sum


def PNNprocess(img, prev): # prev is the index of the previous character
    y1 = model(tf.reshape(img, [1,128,128,1]))

    if prev == 100:
        y2 = np.array([[1/len(chars)] for i in range(len(chars))])
    else:
        y2 = GetProbs(prev)

    return tf.math.multiply(y1, y2)

# Below is the mock code to be used till the NN is up
import matplotlib.pyplot as plt
def PNN_Mock(img, prev): # This is a mock function to test the code until the PNN is up
    img = tf.reshape(img, [128, 128])
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)

    char = input("Enter the character that you just saw. If it could not be distinguised, enter 00 : ")
    if char == "00":
        y1 = [1 / len(chars)] * len(chars)
    else:
        strength = 0.8
        y1 = [(1 - strength) / (len(chars) - 1)] * len(chars)
        y1[chars.index(char)] = strength

    if prev == 100:
        y2 = np.array([[1/len(chars)] for i in range(len(chars))])
    else:
        y2 = GetProbs(prev)

    return tf.math.multiply(y1, y2)


# # Now we apply the logic to get the word recognised
# Create a loop to pass 1 character, then the combination of two characters into the PNN
# Before passing the character into the PNN pass it through pad_image

def CombineImg(image1, image2):
    return pad_image(tf.concat([image1, image2], axis=1), 128, 128)

def ProcessWord(characters):

    word = "" # Add the recognised string successively here

    i = 0
    prev = 100
    while i < len(characters):
        img1 = pad_image(characters[i], 128, 128)
        """The image is processed sideways, it needs to be rotated"""
        y1 = PNN_Mock(tf.reshape(img1, [1,128,128,1]), prev)

        # All p quantities will be length 2 tuples (position of max, value of max)
        p1 = (tf.argmax(y1[0]), tf.math.reduce_max(y1[0]))

        try:
            img2 = CombineImg(characters[i], characters[i + 1])
            y2 = PNN_Mock(tf.reshape(img2, [1,128,128,1]), prev)
            p2 = (tf.argmax(y2[0]), tf.math.reduce_max(y2[0]))
            
        except IndexError:
            p2 = (0, 0)

        if p1[1] > p2[1]:
            word += chars[p1[0]]
        else:
            word += chars[p2[0]]
            i += 1

        i += 1

    return word

y_pred_list = []
label_list = []

for path, label in zip(test_img_paths, test_labels_cleaned):
    cs, label = process_images_labels(path, label)
    label_list.append(label)

    y_pred = ProcessWord(cs)

    print("The word guessed is :", y_pred)
    print("The actual word is ", label)  

    y_pred_list.append(y_pred)  