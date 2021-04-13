#!/usr/bin/python3.6

import math
import os
import random
from enum import Enum

import cv2
import editdistance
from alt_model_checkpoint.keras import AltModelCheckpoint
import tensorflow as tf
import keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, GRU, add, Activation, concatenate, Lambda, \
    BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model, Sequence
from tqdm import tqdm

from concerns.generation_utils import is_image, load_all_image_paths
from model.icr_settings import *


class DataType(Enum):
    TRAIN_DATA = 1
    VALIDATION_DATA = 2
    TEST_DATA = 3


# the actual ctc loss function is defined in the tensorflow backend
def ctc_loss(args):
    y_predict, labels, input_length, label_length = args
    y_predict = y_predict[:, RNN_OUTPUT_START_INDEX:, :]
    return K.ctc_batch_cost(labels,
                            y_predict,
                            input_length,
                            label_length)


def get_input_shape():
    if K.image_data_format() == 'channels_first':
        input_shape = (1, INPUT_HEIGHT, INPUT_WIDTH)
    else:
        input_shape = (INPUT_HEIGHT, INPUT_WIDTH, 1)
    return input_shape


def pre_process(image):
    return image.astype(np.float).reshape(get_input_shape()) / 255.0


def prepare_inputs(data_x, data_y, rnn_time_steps):
    input_lengths = np.repeat(rnn_time_steps - RNN_OUTPUT_START_INDEX,
                              len(data_x))
    input_lengths = np.expand_dims(input_lengths, 1)

    label_lengths = np.array([len(y) for y in data_y], dtype=np.uint8)
    label_lengths = np.expand_dims(label_lengths, 1)

    labels = np.array([np.append(y, [BLANK_CLASS] * (MAX_LABEL_LENGTH - len(y))) for y in data_y],
                      dtype=np.uint8)

    return input_lengths, label_lengths, labels


class DataGenerator(Sequence):
    def __init__(self, files, labels, batch_size, rnn_time_steps, add_blanks=True):
        self.files = files
        self.batch_size = batch_size
        self.add_blanks = add_blanks
        self.rnn_time_steps = rnn_time_steps
        self.labels = labels

        random.shuffle(self.files)

    def __len__(self):
        files_per_batch = self.batch_size - 1 if self.add_blanks else self.batch_size
        return math.ceil(len(self.files) / files_per_batch)

    def __getitem__(self, index):
        input_data = []
        labels = []

        pos = self.batch_size * index
        files_per_batch = self.batch_size - 1 if self.add_blanks else self.batch_size
        for file in self.files[pos:pos + files_per_batch]:
            file_name = os.path.basename(file)
            file_name = file_name[0:file_name.index(os.path.extsep)]

            if file_name.strip() == '':
                continue

            image = pre_process(cv2.imread(file,
                                           cv2.IMREAD_GRAYSCALE))

            label = np.array(LABEL_TO_CLASSES_LAMBDA(self.labels[file_name]),
                             np.uint8)
            input_data.append(image)
            labels.append(label)

        # add a blank
        if self.add_blanks:
            input_data.append(pre_process(np.zeros(shape=(INPUT_HEIGHT,
                                                          INPUT_WIDTH))))
            labels.append(np.array([BLANK_CLASS], np.uint8))

        input_lengths, label_lengths, labels = prepare_inputs(input_data, labels, self.rnn_time_steps)
        return [np.array(input_data), labels, input_lengths, label_lengths], labels


def path_from_data_type(data_type):
    if data_type == DataType.TRAIN_DATA:
        path = TRAIN_DATA_PATH
    elif data_type == DataType.VALIDATION_DATA:
        path = VALIDATION_DATA_PATH
    else:
        path = TEST_DATA_PATH
    return path


def load_file_paths(data_type=DataType.TRAIN_DATA):
    return load_all_image_paths(path_from_data_type(data_type))


def load_labels(data_type=DataType.TRAIN_DATA):
    path = path_from_data_type(data_type)
    labels_path = os.path.join(path, 'labels.dat')
    with open(labels_path, 'rb') as f:
        data = f.read()
    lines = data.split(b'\n')

    labels = dict()
    for l in lines:
        stripped = l.strip()
        if stripped != b'':
            components = stripped.split(b'\0')
            labels[components[0].decode()] = components[1].decode()

    return labels


def sub_sampling(filters, input_layer):
    sub_sample = Conv2D(filters,
                        kernel_size=5,
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(input_layer)
    sub_sample = BatchNormalization()(sub_sample)
    return LeakyReLU(LEAKY_RELU_ALPHA)(sub_sample)


def stacked_convolutions(filters, kernel_size, input_layer, convolutions=2):
    convolution = input_layer
    for i in range(convolutions):
        convolution = Conv2D(filters,
                             (kernel_size, kernel_size),
                             padding='same',
                             kernel_initializer='he_normal')(convolution)
        convolution = BatchNormalization()(convolution)
        convolution = LeakyReLU(LEAKY_RELU_ALPHA)(convolution)

    return convolution


def build_model():
    model_file = MODEL_FILE_NAME + '.h5'
    if os.path.isfile(model_file):
        model = load_model(model_file)
        input_data = None
        predictions = None

        for layer in model.layers:
            if layer.name == 'input_data':
                input_data = layer.output
        for layer in model.layers:
            if layer.name == 'predictions':
                predictions = layer.output
        if input_data is None or predictions is None:
            raise Exception('model does not contain expected layers')

        return input_data, predictions, model, TARGET_RNN_TIME_STEPS

    input_shape = get_input_shape()
    input_data = Input(name='input_data',
                       shape=input_shape)

    embedding = BatchNormalization()(input_data)

    embedding = stacked_convolutions(64 // FILTERS_REDUCTION_FACTOR, 3, embedding)
    if ENABLE_SUB_SAMPLING:
        embedding = sub_sampling(64 // FILTERS_REDUCTION_FACTOR, embedding)
    else:
        embedding = MaxPooling2D(pool_size=(MAX_POOL_SIZE, MAX_POOL_SIZE))(embedding)
    embedding = Dropout(POOLING_DROPOUT_RATE)(embedding)

    embedding = stacked_convolutions(128 // FILTERS_REDUCTION_FACTOR, 3, embedding)
    if ENABLE_SUB_SAMPLING:
        embedding = sub_sampling(128 // FILTERS_REDUCTION_FACTOR, embedding)
    else:
        embedding = MaxPooling2D(pool_size=(MAX_POOL_SIZE, MAX_POOL_SIZE))(embedding)
    embedding = Dropout(POOLING_DROPOUT_RATE)(embedding)

    embedding = stacked_convolutions(256 // FILTERS_REDUCTION_FACTOR, 3, embedding)
    embedding = MaxPooling2D(pool_size=(MAX_POOL_SIZE, MAX_POOL_SIZE))(embedding)
    embedding = Dropout(POOLING_DROPOUT_RATE)(embedding)

    # The exponent (3) is the number of times max-pooling affected the width of the input. The width is reduced by a
    # factor of the size of the max-pooling kernel each time max-pooling is applied. This is the width component of the
    # output tensor of the last convolution layer.
    rnn_time_steps = INPUT_WIDTH // MAX_POOL_SIZE ** 3

    while rnn_time_steps > TARGET_RNN_TIME_STEPS:
        embedding = stacked_convolutions(256 // FILTERS_REDUCTION_FACTOR, 3, embedding)
        embedding = MaxPooling2D(pool_size=(1, MAX_POOL_SIZE))(embedding)
        embedding = Dropout(POOLING_DROPOUT_RATE)(embedding)

        rnn_time_steps //= 2

    last_convolution_filters = 512 // FILTERS_REDUCTION_FACTOR
    embedding = stacked_convolutions(last_convolution_filters, 3, embedding)

    # Reshape the 3-dimensional output of the last convolution layer into a 2-dimensional output such that the width of
    # the output is preserved (which will be used as the number of time-steps of the rnn) and the filters are stacked
    # together into feature vectors for each time-step.
    rnn_input_shape = (TARGET_RNN_TIME_STEPS,
                       INPUT_HEIGHT // MAX_POOL_SIZE ** 3 * last_convolution_filters)

    embedding = Reshape(target_shape=rnn_input_shape)(embedding)

    embedding = Dense(1024, kernel_initializer='he_normal')(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Dropout(EMBEDDING_DROPOUT_RATE)(embedding)
    embedding = LeakyReLU(LEAKY_RELU_ALPHA, name='embedding')(embedding)

    rnn_size = 512
    gru1 = GRU(rnn_size,
               return_sequences=True,
               kernel_initializer='he_normal',
               name='gru1')(embedding)

    backward_gru1 = GRU(rnn_size,
                        return_sequences=True,
                        go_backwards=True,
                        kernel_initializer='he_normal',
                        name='backward_gru1')(embedding)

    gru1_merged = add([gru1, backward_gru1])

    gru2 = GRU(rnn_size,
               return_sequences=True,
               kernel_initializer='he_normal',
               name='gru2')(gru1_merged)

    backward_gru2 = GRU(rnn_size,
                        return_sequences=True,
                        go_backwards=True,
                        kernel_initializer='he_normal',
                        name='backward_gru2')(gru1_merged)

    # the additional class is reserved for the ctc loss blank character
    classes = len(CLASSES) + 1

    # the output of this layer is required for the tensorflow ctc_beam_search_decoder
    predictions = Dense(classes,
                        kernel_initializer='he_normal',
                        name=LOGITS_LAYER_NAME)(concatenate([gru2, backward_gru2]))

    predictions = Activation('softmax',
                             name=PREDICTIONS_LAYER_NAME)(predictions)

    model = Model(inputs=[input_data],
                  outputs=predictions)

    return input_data, predictions, model, TARGET_RNN_TIME_STEPS


def build_ctc_model(input_data, predictions):
    label = Input(name='label',
                  shape=[MAX_LABEL_LENGTH],
                  dtype='float32')

    input_length = Input(name='input_length',
                         shape=[1],
                         dtype='int64')

    label_length = Input(name='label_length',
                         shape=[1],
                         dtype='int64')

    loss_lambda = Lambda(ctc_loss,
                         output_shape=(1,),
                         name='ctc')([predictions, label, input_length, label_length])

    return Model(inputs=[input_data, label, input_length, label_length],
                 outputs=loss_lambda)


def update_report(line='', report_file=MODEL_FILE_NAME + '_report.txt'):
    with open(report_file, 'a') as f:
        f.write(line + '\n')


def train(base_model, ctc_model, rnn_time_steps):
    train_files = load_file_paths(DataType.TRAIN_DATA)
    validation_files = load_file_paths(DataType.VALIDATION_DATA)
    train_labels = load_labels(DataType.TRAIN_DATA)
    validation_labels = load_labels(DataType.VALIDATION_DATA)

    last_model_checkpoint = AltModelCheckpoint(MODEL_FILE_NAME + '_checkpoint.h5',
                                               base_model)
    best_model_checkpoint = AltModelCheckpoint(MODEL_FILE_NAME + '_checkpoint_{epoch:02d}_{val_loss:0.4f}.h5',
                                               base_model,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               mode='min')

    learning_rate_scheduler = LearningRateScheduler(lambda e: EPOCH_LEARNING_RATE_MAP[e], verbose=0)

    update_report('======= training results =======')

    update_report_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: update_report(
        f'epoch: {epoch} - loss: {logs["loss"]} - validation loss: {logs["val_loss"]}'
    ))

    ctc_model.fit_generator(generator=DataGenerator(train_files,
                                                    train_labels,
                                                    BATCH_SIZE,
                                                    rnn_time_steps),
                            steps_per_epoch=math.ceil(len(train_files) / BATCH_SIZE),
                            validation_data=DataGenerator(validation_files,
                                                          validation_labels,
                                                          BATCH_SIZE,
                                                          rnn_time_steps,
                                                          False),
                            validation_steps=math.ceil(len(validation_files) / BATCH_SIZE),
                            epochs=EPOCHS,
                            use_multiprocessing=True,
                            workers=GENERATOR_MAX_WORKERS,
                            max_queue_size=GENERATOR_QUEUE_SIZE,
                            callbacks=[last_model_checkpoint,
                                       best_model_checkpoint,
                                       learning_rate_scheduler,
                                       update_report_callback])

    base_model.save(MODEL_FILE_NAME + '.h5')
    update_report()


def evaluate(base_model, ctc_model, rnn_time_steps):
    test_files = load_file_paths(DataType.TEST_DATA)
    test_labels = load_labels(DataType.TEST_DATA)

    update_report('======= evaluation results =======')

    results = ctc_model.evaluate_generator(generator=DataGenerator(test_files,
                                                                   test_labels,
                                                                   BATCH_SIZE,
                                                                   rnn_time_steps,
                                                                   False),
                                           steps=math.ceil(len(test_files) / BATCH_SIZE),
                                           use_multiprocessing=True,
                                           workers=GENERATOR_MAX_WORKERS,
                                           max_queue_size=GENERATOR_QUEUE_SIZE)

    update_report(f'test loss: {results}')

    evaluate_model(base_model, BATCH_SIZE)


def predict_batch(model, images, greedy=False, merge_repeated=False):
    model_without_softmax = Model(inputs=model.input,
                                  outputs=model.get_layer(LOGITS_LAYER_NAME).output)

    input_tensor = np.array([pre_process(image) for image in images])
    predictions = model_without_softmax.predict(input_tensor)[:, RNN_OUTPUT_START_INDEX:, :]

    time_steps = TARGET_RNN_TIME_STEPS - RNN_OUTPUT_START_INDEX

    decoded_batch = []
    probabilities = []
    for prediction in predictions:
        if greedy:
            decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(prediction.reshape((time_steps, 1, len(CLASSES) + 1)),
                                                               [time_steps])

            prob = tf.reduce_prod(tf.math.reduce_max(tf.nn.softmax(prediction), -1, keepdims=True)).numpy()
            probabilities.append(prob)
        else:
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(prediction.reshape((time_steps, 1, len(CLASSES) + 1)),
                                                              [time_steps])
            probabilities.append(np.exp(log_prob[0]))

        decoded_batch.append(tf.sparse.to_dense(decoded[0]).numpy()[0])

    labels = []
    for seq in decoded_batch:
        if merge_repeated:
            merged = seq
            before = None
            i = 0
            while i < len(merged):
                if merged[i] == before:
                    merged[i] = -1
                    i += 1
                    continue
                before = merged[i]
                i += 1
            merged = list(filter(lambda x: x >= 0, merged))
            labels.append(LABEL_CONCAT_CHAR.join([CLASSES[c] for c in merged]))
        else:
            labels.append(LABEL_CONCAT_CHAR.join([CLASSES[c] for c in seq]))

    return labels, probabilities


def evaluate_model(model, batch_size):
    files = os.listdir(TEST_DATA_PATH)
    files = list(filter(lambda f: is_image(os.path.join(TEST_DATA_PATH, f)), files))
    test_labels = load_labels(DataType.TEST_DATA)

    distances_mean = 0
    distance_ratios_mean = 0
    amount_overall_accuracy = 0

    iterations = math.ceil(len(files) / batch_size)
    progress_bar = tqdm(total=iterations)

    for i in range(iterations):
        batch = files[i * batch_size: (i + 1) * batch_size]
        paths = [os.path.join(TEST_DATA_PATH, f) for f in batch]
        paths = list(filter(lambda x: is_image(x),
                            paths))

        images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        labels = [test_labels[f[0:f.index(os.path.extsep)]] for f in batch]

        predictions, probabilities = predict_batch(model, images)

        for j in range(len(predictions)):
            predicted = predictions[j]
            label = labels[j]

            edit_distance = editdistance.eval(predicted, label)
            distances_mean += edit_distance
            distance_ratios_mean += edit_distance * 1.0 / len(label)

            if label == predicted:
                amount_overall_accuracy += 1
        progress_bar.update()

    progress_bar.close()

    distances_mean /= len(files)
    distance_ratios_mean /= len(files)
    amount_overall_accuracy /= len(files)

    update_report(f'average edit distances: {distances_mean}')
    update_report(f'class recognition error rate: {distance_ratios_mean}')
    update_report(f'class recognition accuracy: {100 - distance_ratios_mean * 100}')
    update_report(f'label recognition error rate: {1 - amount_overall_accuracy}')
    update_report(f'label recognition accuracy: {amount_overall_accuracy * 100}')

    return distances_mean, distance_ratios_mean, amount_overall_accuracy


def run_pipeline():
    input_data, predictions, base_model, rnn_time_steps = build_model()
    base_model.summary()
    plot_model(base_model,
               to_file=MODEL_PLOT_NAME,
               show_shapes=True)

    ctc_model = build_ctc_model(input_data, predictions)
    ctc_model.summary()

    sgd = SGD(lr=INITIAL_LEARNING_RATE,
              decay=0.0,
              momentum=0.9,
              nesterov=True,
              clipnorm=5)

    # the loss function is added in the output of the ctc model so the loss function will be trivial, i.e. returns the
    # same output of the ctc model (y_predict).
    ctc_model.compile(loss={'ctc': lambda y_true, y_predict: y_predict},
                      optimizer=sgd)

    print('\ntraining model...\n')
    train(base_model, ctc_model, rnn_time_steps)

    print('\nevaluating model...\n')
    evaluate(base_model, ctc_model, rnn_time_steps)


if __name__ == '__main__':
    run_pipeline()

    # To try out the model, insert the path to the image and run the following code:
    # _, _, base_model, _ = build_model()
    # image = 255 - cv2.imread('./path/to/image.jpg', cv2.IMREAD_GRAYSCALE)
    # print(predict_batch(base_model, [image]))
