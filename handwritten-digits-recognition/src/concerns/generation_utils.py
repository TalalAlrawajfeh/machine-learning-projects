#!/usr/bin/python3.6

import mimetypes
import multiprocessing
import os
import random
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from multiprocessing import Lock

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_CONTRAST_ALPHA = 1.15
MIN_CONTRAST_ALPHA = 0.5
SPECKLES_PROBABILITY = 0.01
EPSILON = 0.001


# Returns all the extensions for a given mime type.
def get_extensions_for_type(general_type):
    for ext in mimetypes.types_map:
        if mimetypes.types_map[ext].split('/')[0] == general_type:
            yield ext.lower()


# Returns all image extensions, e.g. '.jpg'.
def get_image_extensions():
    return tuple(get_extensions_for_type('image'))


# Returns the extension of a file path.
def file_extension(path):
    return os.path.splitext(os.path.basename(path))[1].lower()


# Returns true if the file extension is an image extension and returns false otherwise.
def is_image(path):
    return os.path.isfile(path) and file_extension(path) in get_image_extensions()


# Returns all the traversed paths within a given directory path that have image extensions.
def load_all_image_paths(base_path):
    image_extensions = get_image_extensions()
    paths = []
    for current_path, sub_dirs, sub_files in os.walk(base_path):
        for f in sub_files:
            path = os.path.join(current_path, f)
            if not os.path.isfile(path) or file_extension(path) not in image_extensions:
                continue
            paths.append(path)
    return paths


# Returns 1 or -1 randomly.
def random_sign():
    return (-1) ** random.randint(0, 9)


# Returns True or False randomly.
def random_bool():
    return True if random_sign() == 1 else False


# Returns a random real number within a range.
def random_within_range(min_random, max_random):
    return (max_random - min_random) * random.random() + min_random


# Returns an image consisting of permuted pixels of the original image either by permuting the rows, permuting the
# columns, or permuting all the pixels at random.
def random_permute_pixels(image):
    perm_type = random.randint(0, 2)
    if perm_type == 0:
        return np.random.permutation(image)
    if perm_type == 1:
        return np.array([np.random.permutation(row) for row in image]).astype(np.uint8)
    return np.random.permutation(np.array([np.random.permutation(row) for row in image]).astype(np.uint8))


# Returns the smallest rectangle that contains all non-zero (gray) pixels.
def find_minimal_bounding_box(image):
    image_height = image.shape[0]
    image_width = image.shape[1]

    y1 = 0
    while y1 < image_height and not np.any(image[y1]):
        y1 += 1
    if y1 == image_width:
        y1 = 0

    y2 = image_height - 1
    while y2 >= 0 and not np.any(image[y2]):
        y2 -= 1
    if y2 == 0:
        y2 = image_height - 1

    temp = image.T

    x1 = 0
    while x1 < image_width and not np.any(temp[x1]):
        x1 += 1
    if x1 == image_width:
        x1 = 0

    x2 = image_width - 1
    while x2 >= 0 and not np.any(temp[x2]):
        x2 -= 1
    if x2 == 0:
        x2 = image_width - 1

    return x1, y1, x2 + 1, y2 + 1


def crop_image(image):
    x1, y1, x2, y2 = find_minimal_bounding_box(image)
    return image[y1:y2, x1:x2]


# Returns a darker or brighter image according to the intensity_delta parameter.
def change_image_intensity(image, intensity_delta):
    image = image.astype(np.int32)
    image[image > 0] += intensity_delta
    image[image > 255] = 255
    image[image < 0] = 0
    return image.astype(np.uint8)


# Takes an inverted image (background intensities are close or equal to zero and foreground intensities are close or
# equal to 255). Blends the image into the background by rescaling the image's pixel intensities range to better match
# the background pixel intensities range, then taking a weighted sum of both. This is done by mapping the minimum pixel
# intensity of the image to a weighted average of the minimum background pixel intensity and the mean background
# pixel intensity (to avoid outliers and get a better estimation of the "minimum intensity of the background");
# and mapping 255 to 255 (to keep the same foreground intensity). All the values in between are mapped linearly.
def blend_images(background, image):
    rescale_value = (np.min(background) + np.mean(background)) / 2

    mask = np.copy(image).astype(np.uint8)
    mask[mask > 0] = 255
    mask[mask == 0] = 1
    mask[mask == 255] = 0

    background_copy = np.copy(background).astype(np.uint8)
    background *= mask
    background_copy *= (1 - mask)

    non_zero_pixels = image[image > 0]
    if len(non_zero_pixels) == 0:
        min_pixel = 0
    else:
        min_pixel = np.min(non_zero_pixels)

    # note that the small value (EPSILON) is added to the denominator to avoid division by zero in some cases
    slope = (255 - rescale_value) / (255 - min_pixel + EPSILON)
    # this is a linear mapping from the pixel range of the second image to the first image
    image[image > 0] = slope * (non_zero_pixels - min_pixel) + rescale_value

    # the result is a weighted sum of the two images
    background += 0.3 * background_copy + 0.7 * image


# Returns the image resized by a factor.
def resize(image, resize_factor=1):
    if resize_factor == 1:
        return image
    resize_width = int(image.shape[1] * resize_factor)
    if resize_width < 2:
        resize_width = 2
    resize_height = int(image.shape[0] * resize_factor)
    if resize_height < 2:
        resize_height = 2
    return cv2.resize(image,
                      (resize_width,
                       resize_height),
                      interpolation=cv2.INTER_AREA)


# Print a given text into a constrained area size. First, the desired_font_size is taken for the initial attempt. If the
# printed text doesn't fit into the area, lower font sizes are tried until the best fit is found. This is done so since
# the actual printed text has empty margins and what is needed is the exact printed text (minimum area containing the
# printed text). Percentages of the background_size_upper_bound are added to it according to the
# safe_margin_upper_bound_percentages parameter because in some cases the printed text may fit perfectly within the
# upper bound; however, because of the empty margins, an overflow occurs.
def print_text_cropped(text,
                       font_path,
                       desired_font_size,
                       foreground_color,
                       background_size_upper_bound,
                       safe_margin_upper_bound_percentages=(1, 1)):
    safe_height = int(background_size_upper_bound[0] * (1 + safe_margin_upper_bound_percentages[0]))
    safe_width = int(background_size_upper_bound[1] * (1 + safe_margin_upper_bound_percentages[1]))
    image = None

    while desired_font_size > 0:
        image = np.zeros((safe_height, safe_width), np.uint8)
        image = Image.fromarray(image, 'L')
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(font_path, desired_font_size)
        width, height = draw.textsize(text, font=font)
        if height > safe_height or width > safe_width:
            desired_font_size -= 1
            continue

        draw.text((0, 0), text, fill=foreground_color, font=font)
        cropped = crop_image(np.asarray(image).astype(np.uint8))

        if cropped.shape[0] < background_size_upper_bound[0] and cropped.shape[1] < background_size_upper_bound[1]:
            image = cropped
            break

        desired_font_size -= 1

    if desired_font_size == 0:
        raise Exception('font size must not be zero')

    return image


# Returns a blurred version of the image or the image itself randomly.
# Note that the blurring algorithm is determined randomly too.
def random_blur(image):
    if random_bool():
        return image

    blur_lambdas = [lambda img: cv2.blur(img, (3, 3)),
                    lambda img: cv2.GaussianBlur(img, (3, 3), 0)]

    return random.choice(blur_lambdas)(image)


# Returns the image with adding gaussian noise or the image itself randomly.
def random_add_gaussian_noise(image, mean=0, variance=1, interval=None):
    if random_bool():
        return image

    width, height = image.shape

    gaussian = np.random.normal(mean,
                                variance,
                                (width, height)).astype(np.float)

    if interval is not None:
        gaussian[gaussian < interval[0]] = interval[0]
        gaussian[gaussian > interval[1]] = interval[1]

    gaussian = image + gaussian
    gaussian[gaussian > 255] = 255
    gaussian[gaussian < 0] = 0

    return gaussian.astype(np.uint8)


# Returns the image with adding speckles by a given probability or the image itself randomly.
def random_add_speckles(image, probability=SPECKLES_PROBABILITY, color_lambda=lambda: 255):
    if random_bool():
        return image

    count = int(probability * image.size)

    xs = np.random.uniform(low=0, high=image.shape[1], size=(count,))
    ys = np.random.uniform(low=0, high=image.shape[0], size=(count,))

    for x, y in zip(xs, ys):
        image[int(y), int(x)] = color_lambda()

    return image


# This is used by the apply_gradient function
class GradientDirection(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


def apply_gradient(image, min_value, max_value, direction=GradientDirection.HORIZONTAL, inverted=False):
    image = image.astype(np.float32)
    grad_sign = random_sign()

    if direction == GradientDirection.HORIZONTAL:
        grad = (max_value - min_value) / image.shape[0]
        i = 0

        def predicate(x):
            return x < image.shape[0]

        def update(x):
            return x + 1

        if inverted:
            i = image.shape[0] - 1

            def predicate(x):
                return x > 0

            def update(x):
                return x - 1

        while predicate(i):
            grad_i = grad * i
            image[i, :] += grad_sign * grad_i
            i = update(i)
    else:
        return apply_gradient(image.T, min_value, max_value, inverted=inverted).T

    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(np.uint8)

    return image


# Returns a modified version of an image by changing pixel intensities and the contrast or returns the image itself.
def random_intensity_contrast_modification(image, intensity_delta, min_intensity_delta=0):
    # each type has 1/3 probability
    modification_type = random.randint(0, 2)
    # type 0 is keeping the image as is.
    if modification_type == 1:
        image = change_image_intensity(image, intensity_delta)
    elif modification_type == 2:
        image = apply_gradient(image,
                               min_intensity_delta,
                               intensity_delta,
                               random.choice([GradientDirection.HORIZONTAL, GradientDirection.VERTICAL]),
                               random_bool())

    if random_bool():
        alpha = random_within_range(MIN_CONTRAST_ALPHA, MAX_CONTRAST_ALPHA)
        image = image.astype(np.float)
        image *= alpha
        image[image > 255] = 255
        image = image.astype(np.uint8)

    return image


# This is used to classify the data into train, test, and validation.
class DataType(Enum):
    TRAIN_DATA = 1
    VALIDATION_DATA = 2
    TEST_DATA = 3


# TODO: flush each block of labels to the labels file (append) to increase performance
# Wraps a function intended for data generation and executes multiple generations in parallel (according to the number
# of cores times 2 since most modern CPUs enable hyper-threading). Also handles storing label information of the data
# and viewing the current progress.
def parallel_data_generation(output_data_count,
                             progress_bar,
                             icr_input_path,
                             start_index,
                             data_generation_function,
                             *function_args):
    progress_lock = Lock()
    labels = []
    indices = list(range(start_index, start_index + output_data_count))

    def wrapped_data_generation_function(*args):
        image, image_label = data_generation_function(*args)

        with progress_lock:
            current_index = random.choice(indices)
            labels.append(str(current_index).encode() + b'\0' + image_label.encode() + b'\n')

            image_name = str(current_index) + '.jpg'
            cv2.imwrite(os.path.join(icr_input_path, image_name),
                        image)

            indices.remove(current_index)
            progress_bar.update()

    while len(indices) > 0:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            for _ in range(len(indices)):
                executor.submit(wrapped_data_generation_function,
                                *function_args)

    labels_file = os.path.join(icr_input_path, 'labels.dat')
    with open(labels_file, 'ab') as f:
        for label in labels:
            f.write(label)
