import math
import random

import numpy as np
import pandas as pd
import tensorflow as tf

EPSILON = 1e-7


def pre_process_data_frame_common(data):
    data.columns = data.columns.str.replace(' ', '')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.sort_values('Timestamp')
    return data.drop('Timestamp', axis=1).drop('Normal/Attack', axis=1).dropna()


def pre_process_training_data_frame(data_cut_start_index, data):
    data = pre_process_data_frame_common(data)

    data = data.loc[data_cut_start_index:]
    data = data.to_numpy().astype(np.float32)

    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    data = (data - min_data) / (max_data - min_data + 1)

    return data, max_data, min_data


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self,
                 data,
                 window_size,
                 output_window_size,
                 batch_size):
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size
        self.number_of_windows = len(data) - window_size - output_window_size
        self.batches = int(math.ceil(self.number_of_windows / batch_size))
        self.indices = list(range(self.number_of_windows))
        self.output_window_size = output_window_size
        random.shuffle(self.indices)

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        batch_indices = self.indices[index: index + self.batch_size]

        input_batch = []
        output_batch = []
        for i in batch_indices:
            input_batch.append(self.data[i: i + self.window_size])
            output_batch.append(self.data[i + self.window_size:i + self.window_size + self.output_window_size])

        return np.array(input_batch, np.float32), np.array(output_batch, np.float32)

    def on_epoch_end(self):
        random.shuffle(self.indices)


def build_bidirectional_gru_forecasting_model(input_window_size, output_window_size, features):
    input_layer = tf.keras.layers.Input((input_window_size, features))

    gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(510, return_sequences=True))(input_layer)
    gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(510, return_sequences=True))(gru)
    dense = tf.keras.layers.Dense(units=510, activation='relu', kernel_initializer='glorot_normal')(gru)
    dense = tf.keras.layers.GlobalAveragePooling1D()(dense)
    dense = tf.keras.layers.Dense(units=features * output_window_size,
                                  activation='relu',
                                  kernel_initializer='glorot_normal')(dense)
    dense = tf.keras.layers.Lambda(lambda tensor: tf.reshape(tensor, (-1, output_window_size, features)))(dense)
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
    return model


def build_tcnn_forecasting_model_v1(input_window_size, output_window_size, features):
    input_layer = tf.keras.layers.Input((input_window_size, features))

    number_of_layers = int(math.ceil(math.log(input_window_size, 2)))
    tcnn = input_layer
    for i in range(number_of_layers):
        tcnn = tf.keras.layers.Conv1D(filters=510,
                                      kernel_size=2,
                                      dilation_rate=2 ** i,
                                      kernel_initializer='glorot_normal',
                                      activation='relu',
                                      padding='causal')(tcnn)

    dense = tf.keras.layers.Dense(units=510, activation='relu', kernel_initializer='glorot_normal')(tcnn)
    dense = tf.keras.layers.GlobalAveragePooling1D()(dense)
    dense = tf.keras.layers.Dense(units=features * output_window_size,
                                  activation='relu',
                                  kernel_initializer='glorot_normal')(dense)
    dense = tf.keras.layers.Lambda(lambda tensor: tf.reshape(tensor, (-1, output_window_size, features)))(dense)
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
    return model


def build_tcnn_forecasting_model_v2(input_window_size, output_window_size, features):
    input_layer = tf.keras.layers.Input((input_window_size, features))

    number_of_layers = int(math.ceil(math.log(input_window_size, 2)))
    tcnn = input_layer
    for i in range(number_of_layers):
        tcnn = tf.keras.layers.Conv1D(filters=510,
                                      kernel_size=2,
                                      dilation_rate=2 ** i,
                                      kernel_initializer='glorot_normal',
                                      activation='relu',
                                      padding='causal')(tcnn)

    tcnn = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(tcnn)
    dense = tf.keras.layers.Dense(units=5100, activation='relu', kernel_initializer='glorot_normal')(tcnn)
    dense = tf.keras.layers.Dense(units=features * output_window_size,
                                  activation='relu',
                                  kernel_initializer='glorot_normal')(dense)
    dense = tf.keras.layers.Lambda(lambda tensor: tf.reshape(tensor, (-1, output_window_size, features)))(dense)
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
    return model


def build_fully_connected_forecasting_model(input_window_size, output_window_size, features):
    input_layer = tf.keras.layers.Input((input_window_size, features))

    dense = tf.keras.layers.Flatten()(input_layer)
    dense = tf.keras.layers.Dense(units=5100,
                                  activation='relu',
                                  kernel_initializer='glorot_normal')(dense)
    dense = tf.keras.layers.Dense(units=5100,
                                  activation='relu',
                                  kernel_initializer='glorot_normal')(dense)
    dense = tf.keras.layers.Dense(features * output_window_size,
                                  activation='relu',
                                  kernel_initializer='glorot_normal')(dense)
    dense = tf.keras.layers.Lambda(lambda tensor: tf.reshape(tensor, (-1, output_window_size, features)))(dense)
    model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
    return model


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestModelCheckpoint, self).__init__()
        self.best_loss = None
        self.best_model = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if self.best_loss is None or np.less(current, self.best_loss):
            self.best_loss = current
            self.best_model = self.model

    def get_best_model(self):
        return self.model


def train_validation_test_split(input_windows,
                                validation_data_percentage,
                                test_data_percentage,
                                window_size):
    data_size = input_windows.shape[0]
    validation_data_size = int(math.floor(data_size * validation_data_percentage))
    test_data_size = int(math.floor(data_size * test_data_percentage))
    training_data_size = data_size - (validation_data_size + test_data_size)

    training_data = input_windows[:training_data_size]
    validation_data = input_windows[training_data_size - window_size:training_data_size + validation_data_size]
    test_data = input_windows[training_data_size + validation_data_size - window_size:]

    return training_data, validation_data, test_data


def max_mean_absolute_error(y_true, y_pred):
    absolute_difference = tf.abs(y_true - y_pred)
    absolute_difference = tf.reshape(absolute_difference, (-1, tf.shape(absolute_difference)[-1]))
    return tf.reduce_max(tf.reduce_mean(absolute_difference, axis=0))


def mean_max_absolute_error(y_true, y_pred):
    absolute_difference = tf.abs(y_true - y_pred)
    absolute_difference = tf.reshape(absolute_difference, (-1, tf.shape(absolute_difference)[-1]))
    return tf.reduce_mean(tf.reduce_max(absolute_difference, axis=0))


def mean_absolute_error(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def max_absolute_error(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))


def custom_loss(y_true, y_pred):
    loss1 = max_mean_absolute_error(y_true, y_pred)
    loss2 = mean_absolute_error(y_true, y_pred)
    loss3 = mean_max_absolute_error(y_true, y_pred)
    return (loss1 + loss2 + loss3) / 3


def train_and_get_best_forecasting_model(forecasting_model,
                                         forecasting_training_sequence,
                                         forecasting_validation_sequence,
                                         training_batch_size,
                                         validation_batch_size,
                                         training_epochs):
    forecasting_model.compile(loss=custom_loss,
                              optimizer=tf.keras.optimizers.SGD(0.002, clipnorm=6.0, momentum=0.9, nesterov=True),
                              metrics=[mean_absolute_error, 
                                       max_mean_absolute_error, 
                                       mean_max_absolute_error,
                                       max_absolute_error])

    forecasting_model.summary()

    best_model_checkpoint = BestModelCheckpoint()
    forecasting_model.fit(forecasting_training_sequence,
                          validation_data=forecasting_validation_sequence,
                          batch_size=training_batch_size,
                          validation_batch_size=validation_batch_size,
                          epochs=training_epochs,
                          callbacks=[best_model_checkpoint])

    best_forecasting_model = best_model_checkpoint.get_best_model()
    best_forecasting_model.save('forecasting_model.h5')

    return best_forecasting_model


def main():
    # model hyper-parameters
    data_cut_start_index = 100000
    input_window_size = 200
    output_window_size = 1
    training_batch_size = 128
    training_epochs = 1000
    validation_data_percentage = 0.1
    test_data_percentage = 0.1
    validation_batch_size = 512
    test_batch_size = 512

    training_data_csv_file = 'SWaT_Dataset_Normal.csv'

    normal_data = pd.read_csv(training_data_csv_file)
    normal_data, max_data, min_data = pre_process_training_data_frame(data_cut_start_index, normal_data)
    features = normal_data.shape[-1]

    training_data, validation_data, test_data = train_validation_test_split(normal_data,
                                                                            validation_data_percentage,
                                                                            test_data_percentage,
                                                                            input_window_size)

    forecasting_training_sequence = DataSequence(training_data, input_window_size, output_window_size,
                                                 training_batch_size)
    forecasting_validation_sequence = DataSequence(validation_data, input_window_size, output_window_size,
                                                   validation_batch_size)
    forecasting_test_sequence = DataSequence(test_data, input_window_size, output_window_size, test_batch_size)

    forecasting_model = build_fully_connected_forecasting_model(input_window_size, output_window_size, features)
    best_forecasting_model = train_and_get_best_forecasting_model(forecasting_model,
                                                                  forecasting_training_sequence,
                                                                  forecasting_validation_sequence,
                                                                  training_batch_size,
                                                                  validation_batch_size,
                                                                  training_epochs)

    result = best_forecasting_model.evaluate(forecasting_test_sequence,
                                             batch_size=test_batch_size)
    print(f'test_mean_absolute_error: {result[0]} - test_max_mean_absolute_error: {result[1]} - mean_max_absolute_error: {result[2]} - max_absolute_error: {result[3]}')


if __name__ == '__main__':
    main()
