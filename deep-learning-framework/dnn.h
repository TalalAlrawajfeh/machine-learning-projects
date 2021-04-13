#ifndef DNN_H
#define DNN_H

#include "matrix.h"

typedef Matrix *(*activation_function_ptr)(Matrix *);
typedef Matrix *(*cost_function_ptr)(Matrix *, Matrix *);

typedef struct _DeepNeuralNetwork
{
    uint hidden_layers;
    uint input_layer_size;
    Matrix *layers_weights;
    Matrix *layers_biases;
    activation_function_ptr *activation_functions;
    activation_function_ptr *gradients;
} DeepNeuralNetwork;

// input is a column vector
Matrix *feed_forward(DeepNeuralNetwork *dnn,
                     Matrix *input);

void train_minibatch(DeepNeuralNetwork *dnn,
                     Matrix *inputs,
                     Matrix *outputs,
                     int batch_size,
                     cost_function_ptr cost_function,
                     cost_function_ptr cost_function_gradient,
                     double learning_rate);

DeepNeuralNetwork *createDNN(uint input_layer_size);

void add_layer(DeepNeuralNetwork *dnn,
               uint neurons,
               activation_function_ptr activation,
               activation_function_ptr grad_activation);

void destroyDNN(DeepNeuralNetwork *dnn);

double relu(double value);

Matrix *m_relu(Matrix *values);

double grad_relu(double value);

double sigmoid(double x);

Matrix *m_sigmoid(Matrix *values);

double grad_sigmoid(double value);

Matrix *mse(Matrix *actual, Matrix *expected);

Matrix *grad_mse(Matrix *actual, Matrix *expected);

Matrix *grad_m_sigmoid(Matrix *values);

Matrix *grad_m_relu(Matrix *values);

Matrix *softmax(Matrix *values);

Matrix *grad_softmax(Matrix *values);

Matrix *cross_entropy_loss(Matrix *actual, Matrix *expected);

Matrix *grad_cross_entropy(Matrix *actual, Matrix *expected);

#endif /* DNN_H */