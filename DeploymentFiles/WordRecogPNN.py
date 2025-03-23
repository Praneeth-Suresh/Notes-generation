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

from PreProcessImage import preprocess_image, CharacterSegmentation, pad_image, DrawBoundaries


root = """data\\"""
img_size = (128, 128)
nn_input = (32, 32)
chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
       'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
       'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
       'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
learning_rate = 0.5
base_path = """C://Work//Programming//Hackathons//Reprose//Notes generation//DeploymentFiles//data//"""
words_list = []
max_length = 10 # This is to limit the code so that it can be run locally
start_number = 150 # This is to allow different segments of the datasets to be accessed

# We get the file names from the datatset with which we can open the images later

count = 0
words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if count < start_number:
        count += 1
        continue
    
    words_list.append(line.strip())

    count += 1
    if count == max_length + start_number:
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

model = tf.keras.Sequential()

model.add(tf.keras.Input(shape=(nn_input[0] * nn_input[1],)))
model.add(layers.Dense(70, activation="sigmoid"))
model.add(layers.Dense(60, activation="sigmoid"))
# model.add(layers.Dense(50, activation="sigmoid"))
# model.add(layers.Dense(50, activation="sigmoid"))
model.add(layers.Dense(60, activation="sigmoid"))
model.add(layers.Dense(len(chars), activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt)

""""In the time that the PNN is being developed, the checkpoints will remain commented.
"""
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(os.getcwd() + "/DeploymentFiles/Checkpoints/" + "PNN").expect_partial()
""" end"""

def GetProbs(prev): # The input is the index of the character, not the character itself

    df2 = pd.read_csv(f"{base_path}/TransitionModel.csv", header=None)

    sum = df2.loc[prev].sum()

    return df2.loc[prev].tolist() / sum

def ResizeImage(img, new_dim=(32,32)):
    img = tf.reshape(img, [img.shape[0], img.shape[1], 1], tf.image.ResizeMethod.GAUSSIAN)
    nn_shape = [new_dim[0], new_dim[1], 1]
    return tf.reshape(tf.image.resize(img, new_dim), nn_shape)

def PNNprocess(img, prev): # prev is the index of the previous character
    y1 = model(tf.reshape(ResizeImage(img, nn_input), [1, nn_input[0] * nn_input[1]]))

    if prev == 100:
        y2 = np.array([1/len(chars) for i in range(len(chars))])
    else:
        y2 = GetProbs(prev)

    return tf.math.multiply(y1, y2)[0]

# Below is the mock code to be used till the NN is up
import matplotlib.pyplot as plt
def PNN_Mock(img, prev): # This is a mock function to test the code until the PNN is up
    img = tf.reshape(img, [128, 128])
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()

    print("Data is to be entered in the format <character><strength> eg: m8 (strength is out of 10)")
    char = input("Enter the character that you just saw. If it could not be distinguised, enter 00 : ")
    if char == "00":
        y1 = [1 / len(chars)] * len(chars)
    else:
        try:
            strength = int(char[1]) / 10
        except (ValueError, IndexError):
            strength = 0.5
            
        y1 = [(1 - strength) / (len(chars) - 1)] * len(chars)
        y1[chars.index(char[0])] = strength

    if prev == 100:
        y2 = np.array([1/len(chars) for i in range(len(chars))])
    else:
        y2 = GetProbs(prev)

    return tf.math.multiply(y1, y2) 


# # Now we apply the logic to get the word recognised
# Create a loop to pass 1 character, then the combination of two characters into the PNN
# Before passing the character into the PNN pass it through pad_image

def CombineImg(image1, image2):
    return tf.concat([image1, image2], axis=1)


def ProcessWord(characters):
    word = "" # Add the recognised string successively here
    segment_final = [0] # Add the final segmentation boundaries here

    i = 0
    prev = 100

    wn = [characters[i].shape[1] for i in range(len(characters))]

    while i < len(characters):

        imgn = [characters[i]]
        yn = [PNN_Mock(tf.reshape(pad_image(imgn[0], 128, 128), [128,128,1]), prev)]
        pn = [(tf.argmax(yn[0]), tf.math.reduce_max(yn[0]))]

        combi = 0 # Numbers of segments in the combination

        while pn[combi - 1][1] <= pn[combi][1]: # This loop will execute once no matter what
            try:
                temp_img = CombineImg(imgn[combi], characters[i + combi + 1])
                imgn.append(temp_img)
                yn.append(PNN_Mock(tf.reshape(pad_image(imgn[-1], 128, 128), [128,128,1]), prev))
                pn.append((tf.argmax(yn[-1]), tf.math.reduce_max(yn[-1])))
            except IndexError:
                pn.append((0,0))

            combi += 1

        temp_img = tf.reshape(tf.reshape(pad_image(imgn[-1], 128, 128), [128,128,1]), [128, 128])
        plt.imshow(temp_img, cmap='gray', vmin=0, vmax=1)
         
        CharRec = chars[pn[combi - 1][0]]
        print("Character recognised as ", CharRec)
        word += CharRec
        # Add the positions of the final segmentation boundaries
        w = [wn[k] for k in range(combi)]
        segment_final.append(sum(w) + segment_final[-1])
        del wn[:combi - 1]

        prev = pn[combi - 1][0].numpy()
        i += combi 

    return word, segment_final


y_pred_list = []
label_list = []

for path, label in zip(test_img_paths, test_labels_cleaned):
    cs, label, image = process_images_labels(path, label)
    label_list.append(label)

    y_pred, segment_final = ProcessWord(cs)

    print("The word guessed is :", y_pred)
    print("The actual word is ", label)  

    DrawBoundaries(segment_final, image)

    y_pred_list.append(y_pred)  