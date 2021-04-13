import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

from concerns.settings import SETTINGS

CLASSES = None
EPOCH_LEARNING_RATE_MAP = None
EPOCHS = None

if len(sys.argv) < 2:
    exit()


def calculate_learning_rates(epoch_learning_rate_mappings):
    epochs = [int(e) for e in epoch_learning_rate_mappings]
    learning_rates = [epoch_learning_rate_mappings[e] for e in epoch_learning_rate_mappings]
    interpolator = PchipInterpolator(epochs, learning_rates)

    x = np.arange(min(epochs), max(epochs) + 1)
    y = interpolator(x)

    epoch_learning_rate_map = dict()
    for i in range(len(x)):
        epoch_learning_rate_map[x[i]] = y[i]

    return epoch_learning_rate_map


# define model parameters
if sys.argv[1] == 'digits':
    CLASSES = SETTINGS['digits_classes']
    MAX_LABEL_LENGTH = SETTINGS['digits_max_length']

    LABEL_CONCAT_CHAR = ''


    def digits_label_to_classes(label):
        return [CLASSES.index(l) for l in label]


    LABEL_TO_CLASSES_LAMBDA = digits_label_to_classes

    INPUT_HEIGHT = SETTINGS['digits_input_image_height']
    INPUT_WIDTH = SETTINGS['digits_input_image_width']

    TRAIN_DATA_PATH = SETTINGS['digits_train_data_path']
    VALIDATION_DATA_PATH = SETTINGS['digits_validation_data_path']
    TEST_DATA_PATH = SETTINGS['digits_test_data_path']

    ENABLE_SUB_SAMPLING = SETTINGS['digits_sub_sampling_enabled']
    FILTERS_REDUCTION_FACTOR = SETTINGS['digits_filters_reduction_factor']
    POOLING_DROPOUT_RATE = SETTINGS['digits_pooling_dropout_rate']
    EMBEDDING_DROPOUT_RATE = SETTINGS['digits_embedding_dropout_rate']
    LEAKY_RELU_ALPHA = SETTINGS['digits_leaky_ReLU_alpha']
    TARGET_RNN_TIME_STEPS = SETTINGS['digits_target_rnn_time_steps']

    EPOCH_LEARNING_RATE_MAP = calculate_learning_rates(SETTINGS['digits_epoch_learning_rate_mappings'])
    BATCH_SIZE = SETTINGS['digits_batch_size']
    GENERATOR_MAX_WORKERS = SETTINGS['digits_generator_max_workers']
    GENERATOR_QUEUE_SIZE = SETTINGS['digits_generator_queue_size']

    MODEL_PLOT_NAME = 'digits.png'
    MODEL_FILE_NAME = 'digits'
else:
    exit()

BLANK_CLASS = len(CLASSES)
# ignore the first two outputs of the rnn
RNN_OUTPUT_START_INDEX = 2
MAX_POOL_SIZE = 2
INITIAL_LEARNING_RATE = EPOCH_LEARNING_RATE_MAP[0]
EPOCHS = max(EPOCH_LEARNING_RATE_MAP)
LOGITS_LAYER_NAME = 'logits'
PREDICTIONS_LAYER_NAME = 'predictions'
