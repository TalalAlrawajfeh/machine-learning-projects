#!/usr/bin/python3.6

from keras.datasets import mnist
from tqdm import tqdm

from concerns.generation_utils import *
from concerns.settings import SETTINGS


def random_resize_factor():
    if random_bool():
        resize_factor = random_within_range(SETTINGS['min_shrink_percentage'],
                                            SETTINGS['max_shrink_percentage'])
    else:
        resize_factor = random_within_range(SETTINGS['min_enlarge_percentage'],
                                            SETTINGS['max_enlarge_percentage'])
    return resize_factor


def random_sequence_length():
    integral_length = random.randint(SETTINGS['digits_min_length'],
                                     SETTINGS['digits_max_length'])
    return integral_length


def random_sequence():
    intervals = []

    for i in range(SETTINGS['digits_min_length'] - 1, SETTINGS['digits_max_length']):
        intervals.append((10 ** i if i > 0 else 0, 10 ** (i + 1) - 1))

    interval = random.choice(intervals)
    return [int(i) for i in str(random.randint(interval[0], interval[1]))]


def resize_to_input_size(image):
    return cv2.resize(image,
                      (SETTINGS['digits_input_image_width'],
                       SETTINGS['digits_input_image_height']),
                      interpolation=cv2.INTER_AREA)


def apply_random_effects(template, noise_mean=0):
    template = random_blur(template)
    template = random_add_gaussian_noise(image=template,
                                         variance=random.randint(SETTINGS['min_foreground_intensity_delta'],
                                                                 SETTINGS['max_foreground_intensity_delta']),
                                         interval=[SETTINGS['min_foreground_intensity_delta'],
                                                   SETTINGS['max_foreground_intensity_delta']],
                                         mean=noise_mean)
    return random_add_speckles(template, color_lambda=lambda: random.randint(0, 255))


def random_background_transform(image):
    resized = resize_to_input_size(image)
    transform_type = random.randint(0, 2)
    if transform_type == 0:
        return resized
    if transform_type == 1:
        return random_permute_pixels(resized)

    return apply_random_effects(default_random_intensity_contrast_modification(resized))


# Returns the image with modifying pixel intensity and contrast or the image itself randomly.
def default_random_intensity_contrast_modification(image):
    return random_intensity_contrast_modification(image,
                                                  random_sign() * random.randint(SETTINGS['min_entire_intensity_delta'],
                                                                                 SETTINGS[
                                                                                     'max_entire_intensity_delta']),
                                                  SETTINGS['min_entire_intensity_delta'])


def default_foreground_random_intensity_modification(image):
    return change_image_intensity(image, random_sign() * random.randint(SETTINGS['min_foreground_intensity_delta'],
                                                                        SETTINGS['max_foreground_intensity_delta']))


def generate_sequence_coordinates(template, image_label_pairs):
    sequence_length = len(image_label_pairs)

    template_height = template.shape[0]
    template_width = template.shape[1]

    x_offsets = []
    y_offsets = []

    sequence_width = 0

    min_y1 = template_height
    max_y2 = 0

    for i in range(sequence_length):
        pair = image_label_pairs[i]
        digit = pair[0]
        label = pair[1]

        digit_height = digit.shape[0]
        digit_width = digit.shape[1]

        y_offset = -digit_height

        if i > 0 and (image_label_pairs[i - 1][1] == '' or label == ''):
            x_offset = random.randint(SETTINGS['enclosing_min_shift_x'],
                                      SETTINGS['enclosing_max_shift_x'])
        else:
            x_offset = random.randint(SETTINGS['digit_min_shift_x'],
                                      SETTINGS['digit_max_shift_x'])

        y_offsets.append(y_offset)
        x_offsets.append(x_offset)

        sequence_width += x_offset + digit_width

        if min_y1 > y_offset:
            min_y1 = y_offset

        y2 = y_offset + digit_height

        if max_y2 < y2:
            max_y2 = y2

    sequence_height = abs(max_y2 - min_y1)
    center_x = (template_width - sequence_width) // 2 - 1
    center_y = (template_height - sequence_height) // 2 - 1
    max_shift_y = (template_height - sequence_height) // 2 - 1

    if center_x < 0:
        center_x = 0

    if center_y < 0:
        center_y = 0

    if max_shift_y < 0:
        max_shift_y = 0

    baseline_x = center_x + random_sign() * random.randint(0, center_x)
    baseline_y = center_y + sequence_height + random_sign() * random.randint(0, max_shift_y)

    sequence_label = ''

    coordinates = []

    digit_x = baseline_x
    for i in range(sequence_length):
        pairs = image_label_pairs[i]
        digit = pairs[0]
        label = pairs[1]

        digit_width = digit.shape[1]

        digit_y = baseline_y + y_offsets[i]
        digit_x += x_offsets[i]

        if digit_y < 0:
            digit_y = 0

        if digit_x < 0:
            digit_x = 0

        if digit_x + digit_width > template_width:
            break

        sequence_label += str(label)
        coordinates.append([digit_x, digit_y])
        digit_x += digit_width

    return coordinates


def generate_handwritten_sequence(classes):
    sequence = random_sequence()
    integral_part_length = len(sequence)
    resize_factor = random_resize_factor()
    image_label_pairs = []

    template = np.zeros((SETTINGS['digits_input_image_height'], SETTINGS['digits_input_image_width']),
                        np.uint8)

    for i in range(integral_part_length):
        digit_label = str(sequence[i])
        digit_image = random.choice(classes[sequence[i]])
        digit_image = crop_image(resize(digit_image, resize_factor))
        image_label_pairs.append((digit_image, digit_label))

    template = template.astype(np.float)

    coordinates = generate_sequence_coordinates(template, image_label_pairs)
    sequence_label = ''

    for i in range(len(image_label_pairs)):
        if i >= len(coordinates):
            break
        coordinate = coordinates[i]
        image, label = image_label_pairs[i]

        image_height = image.shape[0]
        image_width = image.shape[1]

        if image_height + coordinate[1] >= template.shape[0]:
            image_height = template.shape[0] - coordinate[1]

        if image_height <= 0:
            continue

        image = image[:image_height]

        part = template[coordinate[1]:coordinate[1] + image_height, coordinate[0]: coordinate[0] + image_width]
        image = default_foreground_random_intensity_modification(image)

        sequence_label += label
        blend_images(part, image)

    template = default_random_intensity_contrast_modification(template)
    return apply_random_effects(template), sequence_label


def generate_data(data_type=DataType.TRAIN_DATA):
    if data_type == DataType.TRAIN_DATA:
        icr_input_path = SETTINGS['digits_train_data_path']
        output_data_count = SETTINGS['digits_train_data_count']
    elif data_type == DataType.VALIDATION_DATA:
        icr_input_path = SETTINGS['digits_validation_data_path']
        output_data_count = SETTINGS['digits_validation_data_count']
    else:
        icr_input_path = SETTINGS['digits_test_data_path']
        output_data_count = SETTINGS['digits_test_data_count']

    if not os.path.isdir(icr_input_path):
        os.makedirs(icr_input_path)

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    if data_type == DataType.TRAIN_DATA or data_type == DataType.VALIDATION_DATA:
        digit_images = train_images
        digit_labels = train_labels
    else:
        digit_images = test_images
        digit_labels = test_labels

    classes = dict()
    for i in range(len(digit_images)):
        image = digit_images[i]
        label = digit_labels[i]

        if label not in classes:
            classes[label] = []
        classes[label].append(image)

    progress_bar = tqdm(total=output_data_count)
    parallel_data_generation(output_data_count,
                             progress_bar,
                             icr_input_path,
                             0,
                             generate_handwritten_sequence,
                             classes)

    progress_bar.close()


if __name__ == '__main__':
    print("generating icr training data...")
    generate_data()
    print("generating icr validation data...")
    generate_data(DataType.VALIDATION_DATA)
    print("generating icr testing data...")
    generate_data(DataType.TEST_DATA)
