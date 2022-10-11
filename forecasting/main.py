import math
import random
import sys
from datetime import datetime
from typing import Callable, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

EPSILON = 1e-7
tf.config.run_functions_eagerly(False)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    abs_y_true = np.abs(y_true)
    abs_difference = np.abs(y_true - y_pred)
    non_zero_values_mask = np.array(abs_y_true >= EPSILON, np.float32)
    zero_values_mask = np.array(abs_y_true < EPSILON, np.float32)
    abs_y_true_adjusted = (abs_y_true * non_zero_values_mask) + zero_values_mask * 1.0
    return float(np.mean(abs_difference / abs_y_true_adjusted))


def tf_mape(y_true, y_pred):
    abs_y_true = tf.abs(y_true)
    abs_difference = tf.abs(y_true - y_pred)
    non_zero_values_mask = tf.cast(abs_y_true >= EPSILON, tf.float32)
    zero_values_mask = tf.cast(abs_y_true < EPSILON, tf.float32)
    abs_y_true_adjusted = (abs_y_true * non_zero_values_mask) + zero_values_mask * 1.0
    return tf.reduce_mean(abs_difference / abs_y_true_adjusted)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def dot_product(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    return np.sum(array1 * array2)


def sequence_to_feature_vectors(sequence: np.ndarray,
                                feature_vector_length: int) -> np.ndarray:
    number_of_feature_vectors = sequence.shape[0] - feature_vector_length + 1
    feature_vectors = np.zeros((feature_vector_length, number_of_feature_vectors),
                               sequence.dtype)
    for i in range(feature_vector_length):
        feature_vectors[i, :] = sequence[i: i + number_of_feature_vectors]
    return np.transpose(feature_vectors)


def sequence_to_input_output_features(sequence: np.ndarray,
                                      input_feature_vector_length: int,
                                      output_feature_vector_length: int) -> tuple[np.ndarray, np.ndarray]:
    input_feature_vectors = sequence_to_feature_vectors(sequence[:-output_feature_vector_length],
                                                        input_feature_vector_length)
    output_feature_vectors = sequence_to_feature_vectors(sequence[input_feature_vector_length:],
                                                         output_feature_vector_length)
    return input_feature_vectors, output_feature_vectors


def moving_average_centered(sequence: np.ndarray,
                            window_size: int,
                            keep_same_size: bool = True,
                            damp_extremes: bool = False) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError('window_size must odd')
    half_window_size = window_size // 2
    sequence_copy = np.array(sequence, np.float32)
    kernel = np.ones(window_size, np.float32)
    if damp_extremes:
        kernel[0] = 0.5
        kernel[-1] = 0.5
    averages = np.convolve(sequence_copy, kernel, 'valid') / window_size
    if keep_same_size:
        sequence_copy[half_window_size: sequence.shape[0] - half_window_size] = averages
        return sequence_copy
    return averages


def moving_average(sequence: np.ndarray,
                   window_size: int,
                   keep_same_size: bool = True) -> np.ndarray:
    half_window_size = window_size // 2
    if window_size % 2 == 0:
        return moving_average_centered(sequence, half_window_size * 2 + 1, keep_same_size, True)
    else:
        return moving_average_centered(sequence, window_size, keep_same_size, False)


def eliminate_trend_with_moving_average(time_series: np.ndarray,
                                        window_size: int,
                                        keep_same_size: bool = True) -> tuple[np.ndarray, np.ndarray]:
    trend_estimate = moving_average(time_series, window_size, keep_same_size)
    if keep_same_size:
        new_time_series = time_series - trend_estimate
    else:
        half_window_size = window_size // 2
        new_time_series = time_series[half_window_size: time_series.shape[0] - half_window_size] - trend_estimate
    return new_time_series, trend_estimate


def apply_differencing_at(time_series: np.ndarray,
                          order: int,
                          index: int) -> float:
    coefficients = np.array([math.comb(order, i) * (-1) ** i for i in range(order, -1, -1)], np.float32)
    return float(dot_product(time_series[index - order:index + 1], coefficients))


def apply_differencing(time_series: np.ndarray, order: int) -> np.ndarray:
    result = np.zeros(time_series.shape[0] - order, np.float32)
    for i in range(order, time_series.shape[0]):
        result[i - order] = apply_differencing_at(time_series, order, i)
    return result


def apply_seasonal_differencing_at(time_series: np.ndarray,
                                   order: int,
                                   lag: int,
                                   index: int) -> float:
    coefficients = np.array([math.comb(order, i) * (-1) ** i for i in range(order, -1, -1)], np.float32)
    series = np.array([time_series[index - lag * order + lag * i] for i in range(order + 1)])
    return float(dot_product(coefficients, series))


def apply_seasonal_differencing(time_series: np.ndarray,
                                order: int,
                                lag: int) -> np.ndarray:
    result = np.zeros(time_series.shape[0] - order * lag, np.float32)
    for i in range(lag * order, time_series.shape[0]):
        result[i - lag * order] = apply_seasonal_differencing_at(time_series, order, lag, i)
    return result


def apply_lagged_difference(time_series: np.ndarray, lag: int) -> np.ndarray:
    return time_series[lag:] - time_series[:-lag]


def apply_seasonal_smoothing(time_series: np.ndarray,
                             period: int,
                             ma_window_size: int,
                             keep_same_size: bool) -> np.ndarray:
    half_ma_window_size = ma_window_size // 2
    smoothed = moving_average(time_series, ma_window_size, keep_same_size)
    original_series = np.array(time_series)
    if not keep_same_size:
        original_series = original_series[half_ma_window_size:-half_ma_window_size]

    w_k = np.zeros(period, np.float32)
    for k in range(period):
        indices = np.arange(k, original_series.shape[0], period)
        w_k[k] = np.mean(original_series[indices] - smoothed[indices])

    s_k = w_k - np.mean(w_k)
    n = int(math.ceil(original_series.shape[0] / period))
    return np.repeat(np.array([s_k]), n, axis=0).ravel()[:original_series.shape[0]]


def extend_seasonality_to_match_original_series(seasonality: np.ndarray,
                                                time_series_length: int,
                                                period_size: int) -> np.ndarray:
    half_period_size = period_size // 2
    extended_seasonality = np.zeros(time_series_length, np.float32)
    if period_size % 2 != 0:
        extended_seasonality[:half_period_size] = seasonality[half_period_size + 1: 2 * half_period_size + 1]
        extended_seasonality[half_period_size: -half_period_size] = seasonality
        extended_seasonality[-half_period_size:] = seasonality[-half_period_size * 2 - 1: -half_period_size - 1]
    else:
        extended_seasonality[:half_period_size] = seasonality[half_period_size: 2 * half_period_size]
        extended_seasonality[half_period_size: -half_period_size] = seasonality
        extended_seasonality[-half_period_size:] = seasonality[-half_period_size * 2: -half_period_size]
    return extended_seasonality


def extrapolate_trend_to_match_original_series_length(trend_series: np.ndarray,
                                                      trend_fit_evaluation_percentage: float,
                                                      window_size: int) -> np.ndarray:
    half_window_size = window_size // 2
    trend_model, error = find_trend_best_fit(trend_series, trend_fit_evaluation_percentage, 1 + half_window_size)
    series_beginning = trend_model(np.arange(1, 1 + half_window_size).astype(np.float32))
    series_end = trend_model(np.arange(trend_series.shape[0] + half_window_size + 1,
                                       trend_series.shape[0] + 2 * half_window_size + 1).astype(np.float32))
    return np.concatenate([series_beginning, trend_series, series_end])


def evaluate_time_series_model_index_based(model: Callable[[np.ndarray], np.ndarray],
                                           training_data_size: int,
                                           evaluation_data: np.ndarray) -> float:
    predictions = model(np.arange(training_data_size + 1,
                                  training_data_size + 1 + evaluation_data.shape[0]).astype(np.float32))
    return mse(evaluation_data, predictions)


def fit_constant_model(time_series: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    c = np.mean(time_series)
    return lambda x: np.ones(x.shape) * c


def fit_polynomial_model(time_series: np.ndarray,
                         polynomial_degree: int,
                         start_index: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    weights = np.polyfit(np.arange(start_index, time_series.shape[0] + start_index).astype(np.float32),
                         time_series,
                         deg=polynomial_degree)
    return np.poly1d(weights)


def fit_linear_logarithmic_model(time_series: np.ndarray, start_index: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    weights = np.polyfit(np.log(np.arange(start_index, time_series.shape[0] + start_index)).astype(np.float32),
                         time_series,
                         deg=1)
    linear_model = np.poly1d(weights)
    return lambda x: linear_model(np.log(x))


def fit_exponential_model(time_series: np.ndarray, start_index: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    weights = np.polyfit(np.arange(start_index, time_series.shape[0] + start_index).astype(np.float32),
                         np.log(time_series),
                         deg=1)
    linear_model = np.poly1d(weights)
    return lambda x: np.exp(linear_model(x))


def fit_and_evaluate_constant_model(training_data: np.ndarray,
                                    evaluation_data: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray],
                                                                          float]:
    constant_model = fit_constant_model(training_data)
    error = evaluate_time_series_model_index_based(constant_model, training_data.shape[0], evaluation_data)
    return constant_model, error


def fit_and_evaluate_polynomial_model(training_data: np.ndarray,
                                      evaluation_data: np.ndarray,
                                      polynomial_degree: int,
                                      start_index: int = 1) -> tuple[Callable[[np.ndarray], np.ndarray],
                                                                     float]:
    polynomial_model = fit_polynomial_model(training_data, polynomial_degree, start_index)
    error = evaluate_time_series_model_index_based(polynomial_model, training_data.shape[0], evaluation_data)
    return polynomial_model, error


def fit_and_evaluate_linear_logarithmic_model(training_data: np.ndarray,
                                              evaluation_data: np.ndarray,
                                              start_index: int = 1) -> tuple[Callable[[np.ndarray], np.ndarray],
                                                                             float]:
    linear_logarithmic_model = fit_linear_logarithmic_model(training_data, start_index)
    error = evaluate_time_series_model_index_based(linear_logarithmic_model, training_data.shape[0], evaluation_data)
    return linear_logarithmic_model, error


def fit_and_evaluate_exponential_model(training_data: np.ndarray,
                                       evaluation_data: np.ndarray,
                                       start_index: int = 1) -> tuple[Callable[[np.ndarray], np.ndarray],
                                                                      float]:
    exponential_model = fit_exponential_model(training_data, start_index)
    error = evaluate_time_series_model_index_based(exponential_model, training_data.shape[0], evaluation_data)
    return exponential_model, error


def try_or_return(operation: Callable, arguments: Union[list, tuple], return_value: Any):
    try:
        return operation(*arguments)
    except:
        return return_value


def find_trend_best_fit(trend_series: np.ndarray,
                        evaluation_data_percentage: float,
                        fit_start_index: int = 1) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    evaluation_data_size = int(math.floor(trend_series.shape[0] * evaluation_data_percentage))
    training_data = trend_series[:-evaluation_data_size]
    evaluation_data = trend_series[-evaluation_data_size:]

    constant_model_mse_pair = try_or_return(fit_and_evaluate_constant_model,
                                            (training_data, evaluation_data),
                                            (None, None))
    linear_model_mse_pair = try_or_return(fit_and_evaluate_polynomial_model,
                                          (training_data, evaluation_data, 1, fit_start_index),
                                          (None, None))
    quadratic_model_mse_pair = try_or_return(fit_and_evaluate_polynomial_model,
                                             (training_data, evaluation_data, 2, fit_start_index),
                                             (None, None))
    cubic_model_mse_pair = try_or_return(fit_and_evaluate_polynomial_model,
                                         (training_data, evaluation_data, 3, fit_start_index),
                                         (None, None))
    linear_logarithmic_model_mse_pair = try_or_return(fit_and_evaluate_linear_logarithmic_model,
                                                      (training_data, evaluation_data, fit_start_index),
                                                      (None, None))
    exponential_model_mse_pair = try_or_return(fit_and_evaluate_exponential_model,
                                               (training_data, evaluation_data, fit_start_index),
                                               (None, None))
    models = {
        'constant': constant_model_mse_pair,
        'linear': linear_model_mse_pair,
        'quadratic': quadratic_model_mse_pair,
        'cubic': cubic_model_mse_pair,
        'linear_logarithmic': linear_logarithmic_model_mse_pair,
        'exponential': exponential_model_mse_pair
    }

    model_key_with_min_error = None
    min_error = sys.float_info.max

    for model_key in models:
        error = models[model_key][1]
        if error is not None and error < min_error:
            min_error = error
            model_key_with_min_error = model_key

    if model_key_with_min_error == 'constant':
        return fit_constant_model(trend_series), min_error
    if model_key_with_min_error == 'linear':
        return fit_polynomial_model(trend_series, 1, fit_start_index), min_error
    if model_key_with_min_error == 'quadratic':
        return fit_polynomial_model(trend_series, 2, fit_start_index), min_error
    if model_key_with_min_error == 'cubic':
        return fit_polynomial_model(trend_series, 3, fit_start_index), min_error
    if model_key_with_min_error == 'linear_logarithmic':
        return fit_linear_logarithmic_model(trend_series, fit_start_index), min_error
    if model_key_with_min_error == 'exponential':
        return fit_exponential_model(trend_series, fit_start_index), min_error

    return lambda x: np.zeros(x.shape), sys.float_info.max


def eliminate_trend_by_seasonal_smoothing_and_moving_average(time_series: np.ndarray,
                                                             period_size: int,
                                                             trend_fit_evaluation_percentage: float,
                                                             moving_average_window_size: int) -> tuple[np.ndarray,
                                                                                                       np.ndarray,
                                                                                                       np.ndarray]:
    seasonality = apply_seasonal_smoothing(time_series, period_size, period_size, False)
    extended_seasonality = extend_seasonality_to_match_original_series(seasonality, time_series.shape[0], period_size)
    deseasonalized = time_series - extended_seasonality
    smoothed_trend = moving_average(deseasonalized, moving_average_window_size, False)
    extrapolated_trend_series = extrapolate_trend_to_match_original_series_length(smoothed_trend,
                                                                                  trend_fit_evaluation_percentage,
                                                                                  moving_average_window_size)
    residuals = deseasonalized - extrapolated_trend_series
    return residuals, extrapolated_trend_series, extended_seasonality


def eliminate_trend_by_seasonal_smoothing_and_curve_fitting(time_series: np.ndarray,
                                                            period_size: int,
                                                            trend_fit_evaluation_percentage: float) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Callable[[np.ndarray],
             np.ndarray],
    float]:
    seasonality = apply_seasonal_smoothing(time_series, period_size, period_size, False)
    extended_seasonality = extend_seasonality_to_match_original_series(seasonality, time_series.shape[0], period_size)
    deseasonalized = time_series - extended_seasonality

    trend_model, error = find_trend_best_fit(deseasonalized, trend_fit_evaluation_percentage)

    trend_series = trend_model(np.arange(1, deseasonalized.shape[0] + 1).astype(np.float32))
    residuals = deseasonalized - trend_series

    return residuals, trend_series, extended_seasonality, trend_model, error


def eliminate_trend_by_differencing(time_series: np.ndarray,
                                    lag: int,
                                    order: int) -> np.ndarray:
    if lag == 0:
        residuals_with_lagged_difference_trend = np.array(time_series)
    else:
        residuals_with_lagged_difference_trend = apply_lagged_difference(time_series, lag)

    if order == 0:
        lagged_difference_residuals = np.array(residuals_with_lagged_difference_trend)
    else:
        lagged_difference_residuals = apply_differencing(residuals_with_lagged_difference_trend, order)

    return lagged_difference_residuals


def eliminate_trend_and_seasonality_by_differencing(non_seasonal_order, seasonal_lag, seasonal_order, time_series):
    if seasonal_order == 0 or seasonal_lag == 0:
        seasonal_difference_residuals = time_series
    else:
        seasonal_difference_residuals = apply_seasonal_differencing(time_series,
                                                                    seasonal_order,
                                                                    seasonal_lag)
    if non_seasonal_order == 0:
        non_seasonal_difference_residuals = seasonal_difference_residuals
    else:
        non_seasonal_difference_residuals = apply_differencing(seasonal_difference_residuals, non_seasonal_order)

    return non_seasonal_difference_residuals


def sample_autocovariance(time_series: np.ndarray,
                          lag: int) -> float:
    h = abs(lag)
    average = np.mean(time_series)
    sum_of_products = dot_product(time_series[h:] - average, time_series[:time_series.shape[0] - h] - average)
    return sum_of_products / time_series.shape[0]


def sample_autocorrelation(time_series: np.ndarray,
                           lag: int) -> float:
    return sample_autocovariance(time_series, lag) / sample_autocovariance(time_series, 0)


def estimate_error_residuals(time_series: np.array):
    trend_model, _ = find_trend_best_fit(time_series, 0.2)
    trend_series = trend_model(np.arange(1, time_series.shape[0] + 1))
    detrended_series = time_series - trend_series
    new_time_series = detrended_series - np.mean(detrended_series)

    best_window_size = None
    best_residuals = None
    best_residuals_std = None
    for window_size in range(2, time_series.shape[0] // 2):
        seasonality = apply_seasonal_smoothing(new_time_series,
                                               window_size,
                                               window_size,
                                               False)
        extended_seasonality = extend_seasonality_to_match_original_series(seasonality,
                                                                           new_time_series.shape[0],
                                                                           window_size)
        residuals = new_time_series - extended_seasonality
        std_residuals = np.std(residuals)
        if best_residuals_std is None or std_residuals < best_residuals_std:
            best_residuals_std = std_residuals
            best_residuals = residuals
            best_window_size = window_size

    return best_residuals, best_window_size


def find_best_differencing_parameters(time_series: np.ndarray,
                                      lag_values: Union[list[int], tuple[int]],
                                      order_values: Union[list[int], tuple[int]]) -> tuple:
    min_mean_abs_residuals = sys.float_info.max
    best_residuals = None
    best_lag = -1
    best_order = -1
    for lag in lag_values:
        for order in order_values:
            residuals = eliminate_trend_by_differencing(time_series, lag, order)
            mean_abs_residuals = np.mean(np.abs(residuals))
            if mean_abs_residuals < min_mean_abs_residuals:
                min_mean_abs_residuals = mean_abs_residuals
                best_lag = lag
                best_order = order
                best_residuals = residuals
    return best_residuals, best_lag, best_order


def find_seasonal_model_best_differencing_parameters(time_series: np.ndarray,
                                                     seasonal_lag_values: Union[list[int], tuple[int]],
                                                     seasonal_order_values: Union[list[int], tuple[int]],
                                                     non_seasonal_order_values: Union[list[int], tuple[int]]) -> tuple:
    min_std_residuals = sys.float_info.max
    best_residuals = None
    best_seasonal_lag_value = -1
    best_seasonal_order_value = -1
    best_non_seasonal_order_value = -1

    for seasonal_lag in seasonal_lag_values:
        for seasonal_order in seasonal_order_values:
            for non_seasonal_order in non_seasonal_order_values:
                residuals = eliminate_trend_and_seasonality_by_differencing(non_seasonal_order,
                                                                            seasonal_lag,
                                                                            seasonal_order,
                                                                            time_series)
                std_residuals = float(np.std(residuals))
                if std_residuals < min_std_residuals:
                    min_std_residuals = std_residuals
                    best_residuals = residuals
                    best_seasonal_lag_value = seasonal_lag
                    best_seasonal_order_value = seasonal_order
                    best_non_seasonal_order_value = non_seasonal_order

    return best_residuals, best_seasonal_lag_value, best_seasonal_order_value, best_non_seasonal_order_value


def sample_entropy(time_series,
                   window_size,
                   filtering_level):
    series_length = len(time_series)

    windows3 = np.array([time_series[i: i + window_size]
                         for i in range(series_length - window_size)])
    windows2 = np.array([time_series[i: i + window_size]
                         for i in range(series_length - window_size + 1)])
    window_size += 1
    windows1 = np.array([time_series[i: i + window_size]
                         for i in range(series_length - window_size + 1)])

    a = np.sum([np.sum(np.abs(x - windows1).max(axis=1) <= filtering_level) - 1 for x in windows1])
    b = np.sum([np.sum(np.abs(x - windows2).max(axis=1) <= filtering_level) - 1 for x in windows3])

    return -np.log(a / b)


def dates_to_time_series(dates: list[datetime]):
    differences = []

    for i in range(1, len(dates)):
        time_delta = dates[i] - dates[i - 1]
        differences.append(time_delta.days)

    return np.array(differences, np.float32)


def find_fourier_series_with_trend_optimal_parameters(y: np.ndarray,
                                                      period: int,
                                                      number_of_fourier_coefficients: int,
                                                      reg_lambda: float = 0.0,
                                                      start_index: int = 1):
    series_length = y.shape[0]
    i = np.arange(start_index, start_index + series_length).astype(np.float32)

    parameters = []
    for n in range(1, number_of_fourier_coefficients + 1):
        f = 2 * math.pi * n / period
        alpha_n = np.cos(f * i)
        parameters.append(alpha_n)

    for n in range(1, number_of_fourier_coefficients + 1):
        f = 2 * math.pi * n / period
        beta_n = np.sin(f * i)
        parameters.append(beta_n)

    parameters.extend([np.power(i, 0),
                       np.power(i, 1),
                       np.power(i, 2),
                       np.power(i, 3)])

    parameters.append(np.log(i))
    # parameters.append(np.exp(i))

    parameters = np.array(parameters)

    rhs = np.sum(parameters * y, axis=-1)

    lhs_rows = np.expand_dims(parameters, 1)
    lhs_columns = np.expand_dims(parameters, 0)
    lhs = np.sum(lhs_rows * lhs_columns, axis=-1)
    reg_terms = np.eye(parameters.shape[0]) * reg_lambda
    lhs -= reg_terms

    return np.linalg.solve(lhs, rhs)


def fourier_series_with_trend_parameters_to_function(parameters: np.ndarray,
                                                     period: int,
                                                     number_of_fourier_coefficients: int) -> Callable[[np.ndarray],
                                                                                                      np.ndarray]:
    def implicit_function(x):
        coefficients = []
        for n in range(1, number_of_fourier_coefficients + 1):
            f = 2 * math.pi * n / period
            coefficients.append(np.cos(f * x))
        for n in range(1, number_of_fourier_coefficients + 1):
            f = 2 * math.pi * n / period
            coefficients.append(np.sin(f * x))
        coefficients.extend([np.power(x, 0),
                             np.power(x, 1),
                             np.power(x, 2),
                             np.power(x, 3)])
        coefficients.append(np.log(x))
        # coefficients.append(np.exp(x))
        coefficients = np.array(coefficients)

        coefficients_shape = list(range(len(coefficients.shape)))
        coefficients_shape[0], coefficients_shape[1] = coefficients_shape[1], coefficients_shape[0]
        coefficients = np.transpose(coefficients, coefficients_shape)
        return np.sum(coefficients * parameters, axis=-1)

    return implicit_function


class FourierSeriesWithTrendModel:
    def __init__(self, period: int, number_of_fourier_coefficients: int, reg_lambda: float = 0.0):
        self.period = period
        self.number_of_fourier_coefficients = number_of_fourier_coefficients
        self.parameters = None
        self.fourier_model_with_trend = None
        self.reg_lambda = reg_lambda

    def fit(self, x, y):
        try:
            self.parameters = find_fourier_series_with_trend_optimal_parameters(y.ravel(),
                                                                                self.period,
                                                                                self.number_of_fourier_coefficients,
                                                                                self.reg_lambda,
                                                                                np.min(x))
        except:
            self.parameters = np.array([0.0] * (2 * self.number_of_fourier_coefficients + 5))

        self.fourier_model_with_trend = fourier_series_with_trend_parameters_to_function(self.parameters,
                                                                                         self.period,
                                                                                         self.number_of_fourier_coefficients)

    def predict(self, x):
        return self.fourier_model_with_trend(x)


class SARIMAXModel:
    def __init__(self, p, d, q, sp, sd, sq, s, trend_option):
        self.p = p
        self.d = d
        self.q = q
        self.sp = sp
        self.sd = sd
        self.sq = sq
        self.s = s
        self.start_index = None
        self.trend_option = ['n', 'c', 't', 'ct'][trend_option]
        self.sarimax_model = None
        self.sarimax_fitted_model = None

    def fit(self, x, y):
        self.start_index = int(np.min(x))
        try:
            self.sarimax_model = SARIMAX(y,
                                         order=(self.p, self.d, self.q),
                                         seasonal_order=(self.sp, self.sd, self.sq, self.s),
                                         trend=self.trend_option,
                                         validate_specification=False,
                                         mle_regression=False,
                                         enforce_invertibility=False,
                                         enforce_stationarity=False)
            self.sarimax_fitted_model = self.sarimax_model.fit(max_iter=10, disp=False)
        except:
            self.sarimax_model = SARIMAX(y,
                                         order=(0, 0, 0),
                                         seasonal_order=(0, 0, 0, 0),
                                         validate_specification=False,
                                         mle_regression=False,
                                         enforce_invertibility=False,
                                         enforce_stationarity=False,
                                         trend='n')
            self.sarimax_fitted_model = self.sarimax_model.fit(max_iter=1, disp=False)

    def predict(self, x):
        start_index = int(np.min(x))
        end_index = int(np.max(x))
        return self.sarimax_fitted_model.predict(start=start_index - self.start_index + 1,
                                                 end=end_index - self.start_index + 1)


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BestModelCheckpoint, self).__init__()
        self.best_loss = None
        self.best_model = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('tf_mape')
        if self.best_loss is None or np.less(current, self.best_loss):
            self.best_loss = current
            self.best_model = self.model

    def get_best_model(self):
        return self.model

    def get_best_loss(self):
        return self.best_loss


def train(model,
          loss_function,
          metric_function,
          optimizer,
          train_x,
          train_y,
          iterations):
    min_metric_value = None
    best_model = None

    for epoch in range(iterations):
        train_variables = model.trainable_variables
        with tf.GradientTape() as tape:
            prediction = model(train_x, training=True)
            loss_value = loss_function(train_y, prediction)
        metric_value = metric_function(train_y, prediction)
        gradients = tape.gradient(loss_value, train_variables)
        optimizer.apply_gradients(zip(gradients, train_variables))

        if min_metric_value is None or metric_value < min_metric_value:
            min_metric_value = metric_value
            best_model = tf.keras.models.clone_model(model)

    return best_model, min_metric_value


class XGBoostRegressor:
    def __init__(self, parameters):
        self.parameters = parameters
        self.model = None

    def fit(self, x, y):
        self.model = XGBRegressor(**self.parameters)
        self.model.fit(x, y)
        y_pred = self.model.predict(x)
        return mape(y, y_pred)

    def predict(self, x):
        return self.model.predict(x)

    def get_parameters(self):
        return self.parameters


class TCNNModel:
    def __init__(self,
                 feature_vector_length,
                 learning_rate,
                 convolution_filters,
                 iterations,
                 batch_size,
                 noise_factor,
                 random_seed):
        self.feature_vector_length = feature_vector_length
        self.learning_rate = learning_rate
        self.convolution_filters = convolution_filters
        self.iterations = iterations
        self.batch_size = batch_size
        self.dilations = int(math.ceil(math.log(feature_vector_length, 2)))
        self.model = None
        self.noise_factor = noise_factor
        self.random_seed = random_seed

    def fit(self, x, y):
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        with tf.device('/cpu:0'):
            if self.model is None:
                input_layer = tf.keras.layers.Input((self.feature_vector_length, 1))
                sigma = np.mean(x) * self.noise_factor / 2
                tcnn = tf.keras.layers.GaussianNoise(stddev=sigma, seed=self.random_seed)(input_layer)

                for i in range(self.dilations):
                    tcnn = tf.keras.layers.Conv1D(filters=self.convolution_filters,
                                                  kernel_size=2,
                                                  dilation_rate=2 ** i,
                                                  kernel_initializer='glorot_normal',
                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                                                  padding='causal')(tcnn)
                    tcnn = tf.keras.layers.PReLU()(tcnn)

                tcnn = tf.keras.layers.GlobalAveragePooling1D()(tcnn)
                # tcnn = tf.keras.layers.Lambda(lambda tensor: tensor[:, -1])(tcnn)
                dense = tf.keras.layers.Dense(units=1,
                                              kernel_initializer='glorot_normal',
                                              kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                                              activation='relu')(tcnn)
                self.model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
            else:
                for layer in self.model.layers:
                    layer.trainable = True

            self.model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                               optimizer=tf.keras.optimizers.Nadam(self.learning_rate),
                               metrics=tf_mape)

            batch_size = min(x.shape[0], self.batch_size)
            epochs = self.iterations
            batches = int(math.ceil(x.shape[0] / batch_size))
            epochs = int(math.ceil(epochs / batches))
            patience = epochs // 2

            best_model_checkpoint = BestModelCheckpoint()
            self.model.fit(np.expand_dims(x, axis=-1),
                           y,
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=True,
                           # verbose=0,
                           callbacks=[best_model_checkpoint,
                                      tf.keras.callbacks.EarlyStopping(monitor='tf_mape',
                                                                       mode='min',
                                                                       patience=patience)])

            self.model = best_model_checkpoint.get_best_model()
            for layer in self.model.layers:
                layer.trainable = False
            self.model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                               optimizer=tf.keras.optimizers.Adam())
        return best_model_checkpoint.get_best_loss()

    def predict(self, x):
        with tf.device('/cpu:0'):
            return self.model.predict(np.expand_dims(x, axis=-1),
                                      verbose=0)

    def to_json(self):
        return {
            'model': self.model.to_json(),
            'weights': self.model.get_weights(),
            'feature_vector_length': self.feature_vector_length,
            'learning_rate': self.learning_rate,
            'convolution_filters': self.convolution_filters,
            'iterations': self.iterations,
            'batch_size': self.batch_size,
            'noise_factor': self.noise_factor,
            'random_seed': self.random_seed
        }

    @staticmethod
    def from_json(model_json):
        tcnn_model = TCNNModel(model_json['feature_vector_length'],
                               model_json['learning_rate'],
                               model_json['convolution_filters'],
                               model_json['iterations'],
                               model_json['batch_size'],
                               model_json['noise_factor'],
                               model_json['random_seed'])
        tcnn_model.model = tf.keras.models.model_from_json(model_json['model'])
        tcnn_model.model.set_weights(model_json['weights'])
        return tcnn_model


class DataSequence(tf.keras.utils.Sequence):
    def __init__(self,
                 data,
                 input_window_size,
                 output_window_size,
                 batch_size):
        self.data = data
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.batch_size = batch_size
        self.number_of_windows = len(data) - input_window_size - output_window_size + 1
        self.batches = int(math.ceil(self.number_of_windows / batch_size))
        self.indices = list(range(self.number_of_windows))
        # random.shuffle(self.indices)

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        total_windows_sizes = self.input_window_size + self.output_window_size

        input_batch = []
        output_batch = []
        for i in batch_indices:
            input_batch.append(self.data[i: i + self.input_window_size])
            output_batch.append(self.data[i + self.input_window_size:i + total_windows_sizes])

        return np.array(input_batch, np.float32), np.array(output_batch, np.float32)

    def on_epoch_end(self):
        random.shuffle(self.indices)


@tf.function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


class MultivariateMLPModel:
    def __init__(self,
                 input_window_size,
                 output_window_size,
                 features_multiplier,
                 learning_rate,
                 iterations,
                 batch_size,
                 noise_factor,
                 random_seed):
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.learning_rate = learning_rate
        self.features_multiplier = features_multiplier
        self.iterations = iterations
        self.batch_size = batch_size
        self.model = None
        self.features = None
        self.noise_factor = noise_factor
        self.random_seed = random_seed

    def fit(self, time_series):
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.features = time_series.shape[-1]
        with tf.device('/cpu:0'):
            if self.model is None:
                input_layer = tf.keras.layers.Input((self.input_window_size, self.features))
                sigma = np.mean(time_series) * self.noise_factor / 2
                dense_units = self.output_window_size * self.features * self.features_multiplier
                dense = tf.keras.layers.Flatten()(input_layer)
                dense = tf.keras.layers.GaussianNoise(stddev=sigma, seed=self.random_seed)(dense)
                dense = tf.keras.layers.Dense(units=dense_units,
                                              kernel_initializer='glorot_normal')(dense)
                dense = tf.keras.layers.PReLU()(dense)
                dense = tf.keras.layers.Dense(units=dense_units,
                                              kernel_initializer='glorot_normal')(dense)
                dense = tf.keras.layers.PReLU()(dense)
                dense = tf.keras.layers.Dense(self.features * self.output_window_size,
                                              kernel_initializer='glorot_normal')(dense)
                dense = tf.keras.layers.ReLU()(dense)
                dense = tf.keras.layers.Reshape((self.output_window_size, self.features))(dense)
                self.model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
            else:
                for layer in self.model.layers:
                    layer.trainable = True

            self.model.summary()
            self.model.compile(loss=custom_loss,
                               optimizer=tf.keras.optimizers.Nadam(self.learning_rate),
                               metrics=tf_mape)

            batch_size = min(time_series.shape[0], self.batch_size)

            training_sequence = DataSequence(time_series,
                                             self.input_window_size,
                                             self.output_window_size,
                                             batch_size)

            epochs = self.iterations
            batches = len(training_sequence)
            epochs = int(math.ceil(epochs / batches))
            patience = epochs // 2

            best_model_checkpoint = BestModelCheckpoint()
            self.model.fit(training_sequence,
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=True,
                           # verbose=0,
                           callbacks=[best_model_checkpoint,
                                      tf.keras.callbacks.EarlyStopping(monitor='tf_mape',
                                                                       mode='min',
                                                                       patience=patience)])

            self.model = best_model_checkpoint.get_best_model()
            for layer in self.model.layers:
                layer.trainable = False
            self.model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                               optimizer=tf.keras.optimizers.Adam())
        return best_model_checkpoint.get_best_loss()

    def predict(self, x):
        with tf.device('/cpu:0'):
            return self.model.predict(x, verbose=0)

    def to_json(self):
        return {
            'model': self.model.to_json(),
            'weights': self.model.get_weights(),
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'learning_rate': self.learning_rate,
            'features_multiplier': self.features_multiplier,
            'iterations': self.iterations,
            'batch_size': self.batch_size,
            'noise_factor': self.noise_factor,
            'random_seed': self.random_seed
        }

    @staticmethod
    def from_json(model_json):
        mlp_model = MultivariateMLPModel(model_json['input_window_size'],
                                         model_json['output_window_size'],
                                         model_json['features_multiplier'],
                                         model_json['learning_rate'],
                                         model_json['iterations'],
                                         model_json['batch_size'],
                                         model_json['noise_factor'],
                                         model_json['random_seed'])
        mlp_model.model = tf.keras.models.model_from_json(model_json['model'])
        mlp_model.model.set_weights(model_json['weights'])
        return mlp_model


class MultivariateTCNN:
    def __init__(self,
                 input_window_size,
                 output_window_size,
                 features_multiplier,
                 learning_rate,
                 iterations,
                 batch_size,
                 noise_factor,
                 random_seed):
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.learning_rate = learning_rate
        self.features_multiplier = features_multiplier
        self.iterations = iterations
        self.batch_size = batch_size
        self.model = None
        self.features = None
        self.noise_factor = noise_factor
        self.random_seed = random_seed
        self.dilations = int(math.ceil(math.log(self.input_window_size, 2)))

    def fit(self, time_series):
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.features = time_series.shape[-1]
        with tf.device('/cpu:0'):
            if self.model is None:
                input_layer = tf.keras.layers.Input((self.input_window_size, self.features))
                sigma = np.mean(time_series) * self.noise_factor / 2
                convolution_filters = self.output_window_size * self.features * self.features_multiplier
                tcnn = tf.keras.layers.GaussianNoise(stddev=sigma, seed=self.random_seed)(input_layer)
                for i in range(self.dilations):
                    tcnn = tf.keras.layers.Conv1D(filters=convolution_filters,
                                                  kernel_size=2,
                                                  dilation_rate=2 ** i,
                                                  kernel_initializer='glorot_normal',
                                                  kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                                                  padding='causal')(tcnn)
                    tcnn = tf.keras.layers.PReLU()(tcnn)

                tcnn = tf.keras.layers.GlobalAveragePooling1D()(tcnn)
                dense = tf.keras.layers.Dense(units=self.output_window_size * self.features,
                                              kernel_initializer='glorot_normal',
                                              kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3),
                                              activation='relu')(tcnn)
                dense = tf.keras.layers.Reshape((self.output_window_size, self.features))(dense)
                self.model = tf.keras.models.Model(inputs=input_layer, outputs=dense)
            else:
                for layer in self.model.layers:
                    layer.trainable = True

            self.model.summary()
            self.model.compile(loss=custom_loss,
                               optimizer=tf.keras.optimizers.Nadam(self.learning_rate),
                               metrics=tf_mape)

            batch_size = min(time_series.shape[0], self.batch_size)

            training_sequence = DataSequence(time_series,
                                             self.input_window_size,
                                             self.output_window_size,
                                             batch_size)

            epochs = self.iterations
            batches = len(training_sequence)
            epochs = int(math.ceil(epochs / batches))
            patience = epochs // 2

            best_model_checkpoint = BestModelCheckpoint()
            self.model.fit(training_sequence,
                           batch_size=batch_size,
                           epochs=epochs,
                           shuffle=True,
                           # verbose=0,
                           callbacks=[best_model_checkpoint,
                                      tf.keras.callbacks.EarlyStopping(monitor='tf_mape',
                                                                       mode='min',
                                                                       patience=patience)])

            self.model = best_model_checkpoint.get_best_model()
            for layer in self.model.layers:
                layer.trainable = False
            self.model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                               optimizer=tf.keras.optimizers.Adam())
        return best_model_checkpoint.get_best_loss()

    def predict(self, x):
        with tf.device('/cpu:0'):
            return self.model.predict(x, verbose=0)

    def to_json(self):
        return {
            'model': self.model.to_json(),
            'weights': self.model.get_weights(),
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'learning_rate': self.learning_rate,
            'features_multiplier': self.features_multiplier,
            'iterations': self.iterations,
            'batch_size': self.batch_size,
            'noise_factor': self.noise_factor,
            'random_seed': self.random_seed
        }

    @staticmethod
    def from_json(model_json):
        tcnn_model = MultivariateMLPModel(model_json['input_window_size'],
                                          model_json['output_window_size'],
                                          model_json['features_multiplier'],
                                          model_json['learning_rate'],
                                          model_json['iterations'],
                                          model_json['batch_size'],
                                          model_json['noise_factor'],
                                          model_json['random_seed'])
        tcnn_model.model = tf.keras.models.model_from_json(model_json['model'])
        tcnn_model.model.set_weights(model_json['weights'])
        return tcnn_model


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


def normalize_time_series(time_series, normalization_factor=5.0):
    shift = np.min(time_series, axis=0) - 0.001
    scale = np.max(time_series, axis=0) - shift - 0.001
    scale[scale < EPSILON] = 1.0
    scale /= normalization_factor
    return (time_series - shift) / scale, shift, scale


def sliding_window_recursive_multi_step_forecast(time_series: np.ndarray,
                                                 shift: np.ndarray,
                                                 scale: np.ndarray,
                                                 steps: int,
                                                 model,
                                                 input_window_size: int,
                                                 output_window_size: int,
                                                 pre_process_data: bool = True) -> np.ndarray:
    original_history_data = np.array(time_series[-input_window_size:])
    if pre_process_data:
        original_history_data = (original_history_data - shift) / scale

    multi_step_forecast = np.zeros((0, original_history_data.shape[-1]), np.float32)

    current_history_data = original_history_data
    actual_steps = int(math.ceil(steps / output_window_size))
    for i in range(actual_steps):
        next_prediction = model.predict(np.array([current_history_data]))[0]
        multi_step_forecast = np.concatenate([multi_step_forecast, next_prediction], axis=0)
        delta_size = input_window_size - multi_step_forecast.shape[0]
        if delta_size > 0:
            current_history_data = np.concatenate([original_history_data[-delta_size:], multi_step_forecast], axis=0)
        else:
            current_history_data = multi_step_forecast[-input_window_size:]
    return np.array(multi_step_forecast[:steps], np.float32) * scale + shift


def main():
    n = 100

    pattern = [20, 30, 40, 30, 20]
    time_series1 = []

    j = 0
    for i in range(n):
        x = pattern[j] + random.randint(-1, 1)
        x += i * 0.1
        time_series1.append(x)
        j += 1
        if j == len(pattern):
            j = 0

    period = 10
    waves = 20
    coefficients = [random.randint(-5, 5) for _ in range(waves)]
    initial_value = 10
    time_series2 = []
    for i in range(n):
        x = initial_value
        for j in range(waves):
            x += coefficients[j] * math.sin(2 * math.pi * i * j / period)
            x += coefficients[j] * math.cos(2 * math.pi * i * j / period)
        time_series2.append(x)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(time_series1)
    axs[1].plot(time_series2)
    plt.show()

    time_series = np.concatenate([
        np.expand_dims(time_series1, axis=-1),
        np.expand_dims(time_series2, axis=-1)
    ], axis=-1)

    normalized_time_series, shift, scale = normalize_time_series(time_series)

    # ds = DataSequence(time_series,
    #                   5,
    #                   1
    #                   , 64)
    # print(ds[0])
    # print(ds[1])
    # print(time_series)
    # exit()
    plt.plot(normalized_time_series)
    plt.show()

    input_window_size = 30
    output_window_size = 1
    model = MultivariateMLPModel(input_window_size, output_window_size, 100, 0.001, 10000, 64, 0.01, 10000)
    model.fit(normalized_time_series)

    forecast = sliding_window_recursive_multi_step_forecast(time_series,
                                                            shift,
                                                            scale,
                                                            30,
                                                            model,
                                                            input_window_size,
                                                            output_window_size)

    plt.plot(time_series)
    plt.plot(np.arange(len(time_series), len(time_series) + 30), forecast)
    plt.show()


if __name__ == '__main__':
    main()
