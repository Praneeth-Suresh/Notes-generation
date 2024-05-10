import numpy as np
import os
import tensorflow as tf

from PreProcessImage import preprocess_image, CharacterSegmentation

batch_size = 64
image_width, image_height = (512, 128)

# This file implements the training pipeline necessary


# We get the file names from the datatset with which we can open the images later

base_path = """C://Work//Programming//Hackathons//Reprose//Notes generation//DeploymentFiles//data//"""
words_list = []

words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    
    words_list.append(line.strip())

print(len(words_list))

np.random.shuffle(words_list)

# We will split the dataset into three subsets with a 90:5:5 ratio (train:validation:test).

split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_samples = words_list[split_idx:]

val_split_idx = int(0.5 * len(test_samples))
validation_samples = test_samples[:val_split_idx]
test_samples = test_samples[val_split_idx:]

assert len(words_list) == len(train_samples) + len(validation_samples) + len(
    test_samples
)

print(f"Total training samples: {len(train_samples)}")
print(f"Total validation samples: {len(validation_samples)}")
print(f"Total test samples: {len(test_samples)}")

# BUild the data input pipeline

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


train_img_paths, train_labels = get_image_paths_and_labels(train_samples)
validation_img_paths, validation_labels = get_image_paths_and_labels(validation_samples)
test_img_paths, test_labels = get_image_paths_and_labels(test_samples)

def clean_labels(labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels

train_labels_cleaned = clean_labels(train_labels)
validation_labels_cleaned = clean_labels(validation_labels)
test_labels_cleaned = clean_labels(test_labels)

AUTOTUNE = tf.data.AUTOTUNE

def process_images_labels(image_path, label):
    print(image_path)
    image = preprocess_image(image_path)
    # this can produce errors; need to add a try catch here
    return CharacterSegmentation(image, label)

MisSegmented = 0

def prepare_dataset(image_paths, labels):

    global MisSegmented

    char_list = {}

    for path, label in zip(image_paths, labels):
        chars = process_images_labels(path, label)
        if chars != None:
            char_list.update(chars)
        else:
            MisSegmented += 1

    return char_list.keys(), char_list.values() # This is of the form (labels, images)

train_ds_Y, train_ds_X = prepare_dataset(train_img_paths, train_labels_cleaned)
validation_ds_Y, validation_ds_X = prepare_dataset(validation_img_paths, validation_labels_cleaned)
test_ds_Y, test_ds_X = prepare_dataset(test_img_paths, test_labels_cleaned)

