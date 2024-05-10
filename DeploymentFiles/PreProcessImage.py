import keras_ocr
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import transforms
import cv2

from PIL import Image
import numpy as np
import random
import tensorflow_probability as tfp

from sklearn.linear_model import LinearRegression

from skimage.morphology import medial_axis, skeletonize, thin

from skimage.util import invert
import skimage

import math

image_width, image_height = (512, 128)

def GetBox(Pred):
    x_max = 0
    x_min = 10000
    y_max = 0
    y_min = 10000

    for i in Pred[1]:
        x_max = max(i[0], x_max)
        x_min = min(i[0], x_min)
        y_max = max(i[1], y_max)
        y_min = min(i[1], y_min)

    # Each of these 4 variables is an edge of the box
    
    return x_max, x_min, y_max, y_min

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def DisplayWord(word):
    word = tf.reshape(word, (word.shape[0], word.shape[1], 1))
    
    word = tf.image.flip_left_right(word)
    word = tf.transpose(word, perm=[1, 0, 2])
    # word = (word * 255.0).numpy().clip(0, 255).astype(np.uint8)

    plt.imshow(word[:, :, 0], cmap='gray')
    plt.show()

def BgChangeOptimised(word, CleaningCoefficient):

    # The higher the CleaningCoefficient, the more of the dimmer features that are cleaned out

    word = word * CleaningCoefficient

    word = tf.constant(np.where(word == 0, 1, word))

    word = tf.where(word > 1, 1, word)

    word = 1 - word

    # Below, we make the word brighter
    word = word * 1.5

    word = tf.where(word > 1, 1, word)

    DisplayWord(word)

    return tf.constant(word)

def HorizontallHistorgram(image):
    image = tf.reshape(image, [image_width,image_height])
    image = tf.where(image != 0, 1, image)
    
    sums = tf.map_fn(lambda x: tf.math.reduce_sum(x), image)

    dists = [] # This stores the distance between the top and the bottom pixel

    for i in range(len(image)):
        non_zero = tf.where(image[i] != 0)
        
        if non_zero.shape == (0,1):
            dists.append(0)
        else:
            top = non_zero[0][0].numpy()
            bottom = non_zero[-1][0].numpy()
            dists.append(bottom - top + 1)

    dists = tf.constant(dists, dtype=tf.float32)

    Hw = tf.divide(sums, dists)

    return tf.where(tf.math.logical_not(tf.math.is_finite(Hw)), tf.zeros_like(Hw), Hw)

    # return sums, dists

def VerticalHistorgram(image):
    image = tf.reshape(image, [image_width,image_height])
    sums = tf.reduce_sum(image, axis=0)
    return sums

def Remove_Slope(image):
    def RotateImage(image, angle):
        image = tf.reshape(image, [image_width,image_height])
        im = image.numpy()
        im = Image.fromarray(im)
        im_rotated = im.rotate(angle=angle)
        im = np.asarray(im_rotated)
        image_r = tf.convert_to_tensor(im)

        return image_r

    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)

    Vals = VerticalHistorgram(image)

    def GetVariance(Vals, threshold):
        below_threshold_mask = tf.math.less_equal(Vals, threshold)
        weight_below = tf.math.reduce_sum(tf.cast(below_threshold_mask, dtype=tf.float32))
        vals_below = tf.cast(below_threshold_mask, dtype=tf.float32) * Vals # These are the values below the threshold
        vals_below = tf.boolean_mask(vals_below, tf.math.logical_not(tf.math.equal(vals_below, 0))) # Remove the zeros
        variance_below_threshold = tf.math.reduce_variance(vals_below)

        above_threshold_mask = tf.math.greater(Vals, threshold)
        weight_above = tf.math.reduce_sum(tf.cast(above_threshold_mask, dtype=tf.float32))
        vals_above = tf.cast(above_threshold_mask, dtype=tf.float32) * Vals # These are the values above the threshold
        vals_above = tf.boolean_mask(vals_above, tf.math.logical_not(tf.math.equal(vals_above, 0))) # Remove the zeros
        variance_above_threshold = tf.math.reduce_variance(vals_above)

        weighted_variance = ( variance_above_threshold * weight_above + variance_below_threshold * weight_below ) / ( weight_above + weight_below )

        return weighted_variance

    x = []
    y = []

    for i in range(5, int(tfp.stats.percentile(Vals, 90)), 1):
        x.append(i)

        y.append(GetVariance(Vals, i))

    try:
        core_threshold = x[y.index(min(y))]
    except ValueError:
        return image

    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)

    above_threshold_mask = tf.math.greater(Vals, core_threshold)
    locs = tf.where(above_threshold_mask)
    core_region_base = int(locs[-1])
    core_width = len(locs)

    low_limit = int(core_region_base - 0.25 * core_width)
    upper_limit = int(core_region_base + 0.25 * core_width)

    if upper_limit > image_height:
        upper_limit = image_height - 1

    # This attempts to find the bases of characters within the base character bounds
    # This has significant run-time

    base = []

    for column in range(len(image)):
        found = False # To reset
        index = 0
        val_max = 0 # To reset

        for i in range(low_limit, upper_limit):
            # print(image[column][i])
            if image[column][i] != 0 and found == False:
                found = True
                if image[column][i] > val_max:
                    index = i
                    val_max = image[column][i]

            elif image[column][i] != 0 and found == True:
                if image[column][i] > val_max:
                    index = i
                    val_max = image[column][i]        

            elif image[column][i] == 0 and found == True:   
                found = False
                val_max = 0

        if image[column][upper_limit] == 0: # This prevents lines which extend below from being counted
            base.append(index)
        else:
            base.append(0)

    line = 0
    found = False # To reset
    index = 0
    val_max = 0
    val_max_list = []
    val_max_indices = []

    for i in range(len(base)):
        if base[i] != 0 and found == False:
            line += 1
            found = True
            if base[i] > val_max:
                index = i
                val_max = base[i]

        elif base[i] != 0 and found == True:
            if base[i] > val_max:
                index = i
                val_max = base[i]        

        elif base[i] == 0 and found == True:   
            found = False
            val_max_list.append(val_max)
            val_max_indices.append(index)
            val_max = 0

    val_max_indices = tf.reshape(tf.constant(val_max_indices), [len(val_max_indices),1])
    val_max_list =  tf.reshape(tf.constant(val_max_list), [len(val_max_list),1])

    line = LinearRegression().fit(val_max_indices, val_max_list)

    rotate = - math.atan(line.coef_[0][0]) * 180 / math.pi
    image = RotateImage(image, rotate)

    return image


def remove_slant(image):
    # Shear investigation function
    def ShearTest2(image, angle, reps):
        # When reps = 1, the function takes around 1.5 s to run

        scores = []

        image = tf.reshape(image, [image_width, image_height])
        image = tf.reshape(image, [1, image_width, image_height])

        for rep in range(reps):

            InitAngle = rep * angle

            shear = tf.keras.preprocessing.image.apply_affine_transform(image, theta = InitAngle, shear = -InitAngle)
            
            Hist = HorizontallHistorgram(shear)

            Score = tf.math.reduce_sum(tf.math.square(Hist))

            scores.append(Score)
        
        scores = tf.stack(scores)
        return scores


    def GetShearAngle(image):
        rep = 20
        max_angle = 10

        angle = max_angle / rep

        positive = ShearTest2(image, angle, rep)
        negative = ShearTest2(image, - angle, rep)

        score = tf.concat([negative, positive], 0)

        sorted_indices = tf.argsort(score)
        index_min = sorted_indices[2].numpy()

        shear_angle = ( index_min - rep ) * angle

        print(f"Shear angle : {shear_angle}")

        return shear_angle

    ShearAngle = GetShearAngle(image)

    image = tf.reshape(image, [image_width, image_height])
    image = tf.reshape(image, [1, image_width, image_height])

    shear = tf.keras.preprocessing.image.apply_affine_transform(image, theta = ShearAngle, shear = - ShearAngle)

    image = shear

    image = tf.reshape(image, [image_width, image_height])

    return image





def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    
    # Black background
    def ClCoeff(image):

        nonzero_indices = tf.where(tf.not_equal(tf.math.reduce_sum(image, axis=1), 0))
        first_nonzero_index = tf.reduce_min(nonzero_indices, axis=0)[0]
        last_nonzero_index = tf.reduce_max(nonzero_indices, axis=0)[0]

        return 1 / tf.math.reduce_mean(image[first_nonzero_index: last_nonzero_index])

    image = BgChangeOptimised(image, ClCoeff(image))

    # Slope removal
    image = Remove_Slope(image)

    # Slant removal
    image = remove_slant(image)

    return image





def CharacterSegmentation(image, label):
    # All this is for the purpose of visualising the points at which the breaks needed to be drawn

    def DrawVertical(x_position, image):

        height, width = image.shape
        mask = tf.zeros_like(image)
        mask = tf.concat([image[:x_position, :], tf.ones([1, width]), image[x_position+1:, :]], axis=0)  # White vertical line

        return mask

    def DrawBoundaries(positions, image):
        LinedImg = DrawVertical(positions[0], image)

        for i in range(1, len(positions)):
            LinedImg = DrawVertical(positions[i], LinedImg)

        DisplayWord(LinedImg)

        return LinedImg

    image = tf.reshape(image, [image.shape[0], image.shape[1]])

    TempImg = tf.math.square(image)

    sums = tf.reduce_sum(TempImg, axis=1)
    positions = tf.range(0, image_width, dtype=tf.float32)

    def LossFnTree(begin, end, split):

        local_segments = [split, end]

        VarTotal = 0

        prev = begin

        Averages = []

        for i in range(len(local_segments)):

            Avg = tf.math.reduce_sum(tf.math.multiply(sums[prev:local_segments[i]], positions[prev:local_segments[i]])) / tf.math.reduce_sum(sums[prev:local_segments[i]])

            VarTotal += tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(tf.math.square(positions[prev:local_segments[i]] - Avg), sums[prev:local_segments[i]])) / tf.math.reduce_sum(sums[prev:local_segments[i]]))

            Averages.append(Avg)

            prev = local_segments[i]

        if math.isnan(VarTotal.numpy()):
            loss0 = 0
        else:
            loss0 = VarTotal.numpy()

        if math.isnan(tf.math.reduce_variance(tf.stack(Averages)).numpy()):
            loss1 = 0
        else:
            loss1 = tf.math.reduce_variance(tf.stack(Averages)).numpy()

        if math.isnan(abs((end - split) - (split - begin))):
            loss2 = 0
        else:
            loss2 = abs((end - split) - (split - begin))

        return [loss0, loss1, loss2]

    def Normalise(l, m): # Get the value to between 1 and m
        factor = max(l) / m

        try:
            return [ e/factor for e in l]
        except ZeroDivisionError:
            return l
        
    def TreeSegmentation(segments, threshold):    
        differences = [segments[i + 1] - segments[i] for i in range(len(segments) - 1)]

        while True:

            if sum([True if j < threshold else False for j in differences]) == 1:
                index = -2
            elif sum([True if j < threshold else False for j in differences]) == 0:
                index = 0
            else:
                break

            begin = segments[index]
            end = segments[index + 1]

            losses = []

            for split in range(begin, end):
                losses.append(LossFnTree(begin, end, split))

            loss0 = [i[0] for i in losses]
            loss1 = [i[1] for i in losses]
            loss2 = [i[2] for i in losses]

            loss0 = Normalise(loss0, 10)
            loss1 = Normalise(loss1, 15)
            loss2 = Normalise(loss2, 5)

            losses = [ loss0[i] - loss1[i] + loss2[i] for i in range(len(losses))]

            # plt.plot(losses)
            # plt.show()

            segments.append(losses.index(min(losses)) + begin)

            segments.sort()

            print(segments)

            differences = [segments[i + 1] - segments[i] for i in range(len(segments) - 1)]

        DrawBoundaries(segments, image)

        return segments, differences

    segments, differences = TreeSegmentation([tf.argmax(tf.cast(tf.not_equal(sums, 0), tf.int32), output_type=tf.int32).numpy(), image_width - tf.argmax(tf.cast(tf.not_equal(tf.reverse(sums, axis=[0]), 0), tf.int32), output_type=tf.int32).numpy()], threshold = int(image_width / 50))

    avg_diff = sum(differences) / ( len(differences) + 2 )

    new_segments = []

    for i in range(len(differences)):
        if differences[i] > avg_diff:
            n, d = TreeSegmentation([segments[i], segments[i + 1]], threshold = avg_diff)
            new_segments += n

    boundaries = list(set(new_segments + segments))

    DrawBoundaries(boundaries, image)

    boundaries.sort()
    left = min(boundaries)
    image = image[left:max(boundaries)]
    boundaries = [i - left for i in boundaries]

    characters = []

    for i in range(len(boundaries) - 1):
        if abs(boundaries[i] - boundaries[i + 1]) > 3:
            char = image[boundaries[i]: boundaries[i + 1]]
            print(char.shape)
            characters.append(char)

    if len(characters) == len(label):
        labelled_characters = {}
        for char_img, char in zip(characters, label):
            labelled_characters.update({label: char_img})

        return labelled_characters

    return None    