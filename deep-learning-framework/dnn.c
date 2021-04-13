#include "dnn.h"
#include <stdlib.h>

double special_rand();

// input is a column vector
Matrix *feed_forward(DeepNeuralNetwork *dnn,
                     Matrix *input)
{
    Matrix *weights = &dnn->layers_weights[0];
    Matrix *biases = &dnn->layers_biases[0];

    Matrix *product = matrix_product(weights, input);
    Matrix *weighted_sum = add_matrices(product, biases);

    deallocate_matrix(product);
    product = NULL;

    Matrix *activations = dnn->activation_functions[0](weighted_sum);

    deallocate_matrix(weighted_sum);
    weighted_sum = NULL;

    for (uint layer = 1; layer < dnn->hidden_layers; layer++)
    {
        weights = &dnn->layers_weights[layer];
        biases = &dnn->layers_biases[layer];

        product = matrix_product(weights, activations);

        deallocate_matrix(activations);
        activations = NULL;

        weighted_sum = add_matrices(product, biases);

        deallocate_matrix(product);
        product = NULL;

        activations = dnn->activation_functions[layer](weighted_sum);

        deallocate_matrix(weighted_sum);
        weighted_sum = NULL;
    }

    return activations;
}

void train_minibatch(DeepNeuralNetwork *dnn,
                     Matrix *inputs,
                     Matrix *outputs,
                     int batch_size,
                     cost_function_ptr cost_function,
                     cost_function_ptr cost_function_gradient,
                     double learning_rate)
{
    Matrix **layers_weighted_sums = (Matrix **)malloc(sizeof(Matrix *) * dnn->hidden_layers * batch_size);
    Matrix **layers_activations = (Matrix **)malloc(sizeof(Matrix *) * dnn->hidden_layers * batch_size);
    Matrix **errors = (Matrix **)malloc(sizeof(Matrix *) * dnn->hidden_layers * batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        // feed forward
        Matrix *input = &inputs[i];
        Matrix *weights = &dnn->layers_weights[0];
        Matrix *biases = &dnn->layers_biases[0];

        Matrix *product = matrix_product(weights, input);
        Matrix *weighted_sum = add_matrices(product, biases);

        deallocate_matrix(product);
        product = NULL;

        Matrix *activations = dnn->activation_functions[0](weighted_sum);

        layers_weighted_sums[i * dnn->hidden_layers + 0] = weighted_sum;
        layers_activations[i * dnn->hidden_layers + 0] = activations;

        int layer;
        for (layer = 1; layer < dnn->hidden_layers; layer++)
        {
            weights = &dnn->layers_weights[layer];
            biases = &dnn->layers_biases[layer];

            product = matrix_product(weights, activations);
            weighted_sum = add_matrices(product, biases);

            deallocate_matrix(product);
            product = NULL;

            activations = dnn->activation_functions[layer](weighted_sum);

            layers_weighted_sums[i * dnn->hidden_layers + layer] = weighted_sum;
            layers_activations[i * dnn->hidden_layers + layer] = activations;
        }

        // backpropagation
        layer = dnn->hidden_layers - 1;

        Matrix *gradient = dnn->gradients[layer](layers_weighted_sums[i * dnn->hidden_layers + layer]);
        Matrix *gradient_T = transpose(gradient);
        Matrix *cost_gradient = cost_function_gradient(layers_activations[i * dnn->hidden_layers + layer], &outputs[i]);
        Matrix *error = matrix_product(gradient_T, cost_gradient);

        deallocate_matrix(gradient);
        deallocate_matrix(gradient_T);
        deallocate_matrix(cost_gradient);
        errors[i * dnn->hidden_layers + layer] = error;

        for (layer = layer - 1; layer >= 0; layer--)
        {
            gradient = dnn->gradients[layer](layers_weighted_sums[i * dnn->hidden_layers + layer]);
            gradient_T = transpose(gradient);
            Matrix *weights_T = transpose(&dnn->layers_weights[layer + 1]);
            Matrix *temp = matrix_product(weights_T, errors[i * dnn->hidden_layers + layer + 1]);
            error = matrix_product(gradient_T, temp);

            deallocate_matrix(gradient);
            deallocate_matrix(gradient_T);
            deallocate_matrix(weights_T);
            deallocate_matrix(temp);

            errors[i * dnn->hidden_layers + layer] = error;
        }
    }

    // update dnn
    // update biases
    Matrix *biases = &dnn->layers_biases[0];

    Matrix *sum = errors[0 * batch_size + 0];
    for (uint i = 1; i < batch_size; i++)
    {
        Matrix *temp = add_matrices(sum, errors[i * dnn->hidden_layers + 0]);
        if (i > 1)
            deallocate_matrix(sum);
        sum = temp;
    }

    Matrix *biases_update = multiply_scalar(sum, learning_rate);
    deallocate_matrix(sum);
    sum = biases_update;
    biases_update = multiply_scalar(biases_update, 1.0 / batch_size);
    deallocate_matrix(sum);

    Matrix *new_biases = subtract_matrices(biases, biases_update);

    deallocate_matrix(biases_update);

    // TODO: may be this could be done better
    free(biases->data);
    biases->data = new_biases->data;
    free(new_biases);

    // update weights
    Matrix *weights = &dnn->layers_weights[0];
    Matrix *input_T = transpose(&inputs[0]);
    sum = matrix_product(errors[0 * batch_size + 0], input_T);
    deallocate_matrix(input_T);

    for (uint i = 1; i < batch_size; i++)
    {
        input_T = transpose(&inputs[i]);
        Matrix *prod = matrix_product(errors[i * dnn->hidden_layers + 0], input_T);
        Matrix *temp = add_matrices(sum, prod);
        deallocate_matrix(sum);
        deallocate_matrix(prod);
        sum = temp;
    }

    Matrix *weights_update = multiply_scalar(sum, learning_rate);
    deallocate_matrix(sum);
    sum = weights_update;
    weights_update = multiply_scalar(weights_update, 1.0 / batch_size);
    deallocate_matrix(sum);

    Matrix *new_weights = subtract_matrices(weights, weights_update);

    deallocate_matrix(weights_update);

    // TODO: may be this could be done better
    free(weights->data);
    weights->data = new_weights->data;
    free(new_weights);

    for (uint layer = 1; layer < dnn->hidden_layers; layer++)
    {
        biases = &dnn->layers_biases[layer];

        sum = errors[0 * batch_size + layer];
        for (uint i = 1; i < batch_size; i++)
        {
            Matrix *temp = add_matrices(sum, errors[i * dnn->hidden_layers + layer]);
            if (i > 1)
                deallocate_matrix(sum);
            sum = temp;
        }

        biases_update = multiply_scalar(sum, learning_rate);
        deallocate_matrix(sum);
        sum = biases_update;
        biases_update = multiply_scalar(biases_update, 1.0 / batch_size);
        deallocate_matrix(sum);

        new_biases = subtract_matrices(biases, biases_update);

        deallocate_matrix(biases_update);

        // TODO: may be this could be done better
        free(biases->data);
        biases->data = new_biases->data;
        free(new_biases);

        weights = &dnn->layers_weights[layer];
        input_T = transpose(layers_activations[0 * batch_size + layer - 1]);
        sum = matrix_product(errors[0 * batch_size + layer], input_T);
        deallocate_matrix(input_T);

        for (uint i = 1; i < batch_size; i++)
        {
            input_T = transpose(layers_activations[i * dnn->hidden_layers + layer - 1]);
            Matrix *prod = matrix_product(errors[i * dnn->hidden_layers + layer], input_T);
            Matrix *temp = add_matrices(sum, prod);
            deallocate_matrix(sum);
            deallocate_matrix(prod);
            sum = temp;
        }

        weights_update = multiply_scalar(sum, learning_rate);
        deallocate_matrix(sum);
        sum = weights_update;
        weights_update = multiply_scalar(weights_update, 1.0 / batch_size);
        deallocate_matrix(sum);

        new_weights = subtract_matrices(weights, weights_update);

        deallocate_matrix(weights_update);

        // TODO: may be this could be done better
        free(weights->data);
        weights->data = new_weights->data;
        free(new_weights);
    }

    for (uint i = 0; i < batch_size * dnn->hidden_layers; i++)
    {
        free(layers_activations[i]);
        free(layers_weighted_sums[i]);
        free(errors[i]);
    }

    free(layers_activations);
    free(layers_weighted_sums);
    free(errors);
}

DeepNeuralNetwork *createDNN(uint input_layer_size)
{
    DeepNeuralNetwork *dnn = (DeepNeuralNetwork *)malloc(sizeof(DeepNeuralNetwork));

    dnn->hidden_layers = 0;
    dnn->input_layer_size = input_layer_size;
    dnn->layers_biases = NULL;
    dnn->layers_weights = NULL;
    dnn->activation_functions = NULL;
    dnn->gradients = NULL;

    return dnn;
}

void add_layer(DeepNeuralNetwork *dnn,
               uint neurons,
               activation_function_ptr activation,
               activation_function_ptr grad_activation)
{
    dnn->hidden_layers++;

    if (dnn->layers_biases == NULL)
    {
        dnn->layers_biases = (Matrix *)malloc(sizeof(Matrix));
        dnn->layers_biases[0].columns = 1;
        dnn->layers_biases[0].rows = neurons;
        dnn->layers_biases[0].data = (double *)malloc(sizeof(double) * neurons);
        for (int i = 0; i < neurons; i++)
        {
            dnn->layers_biases[0].data[i] = special_rand();
        }

        dnn->layers_weights = (Matrix *)malloc(sizeof(Matrix));
        dnn->layers_weights[0].columns = dnn->input_layer_size;
        dnn->layers_weights[0].rows = neurons;
        dnn->layers_weights[0].data = (double *)malloc(sizeof(double) * dnn->input_layer_size * neurons);
        for (int i = 0; i < dnn->input_layer_size * neurons; i++)
        {
            dnn->layers_weights[0].data[i] = special_rand();
        }

        dnn->activation_functions = (activation_function_ptr *)malloc(sizeof(activation_function_ptr));
        dnn->activation_functions[0] = activation;

        dnn->gradients = (activation_function_ptr *)malloc(sizeof(activation_function_ptr));
        dnn->gradients[0] = grad_activation;
    }
    else
    {
        int pos = dnn->hidden_layers - 1;
        dnn->layers_biases = (Matrix *)realloc(dnn->layers_biases, sizeof(Matrix) * dnn->hidden_layers);
        dnn->layers_biases[pos].columns = 1;
        dnn->layers_biases[pos].rows = neurons;
        dnn->layers_biases[pos].data = (double *)malloc(sizeof(double) * neurons);
        for (int i = 0; i < neurons; i++)
        {
            dnn->layers_biases[pos].data[i] = special_rand();
        }

        dnn->layers_weights = (Matrix *)realloc(dnn->layers_weights, sizeof(Matrix) * dnn->hidden_layers);
        dnn->layers_weights[pos].columns = dnn->layers_weights[pos - 1].rows;
        dnn->layers_weights[pos].rows = neurons;
        dnn->layers_weights[pos].data = (double *)malloc(sizeof(double) * dnn->layers_weights[pos].columns * neurons);
        for (int i = 0; i < dnn->layers_weights[pos].columns * neurons; i++)
        {
            dnn->layers_weights[pos].data[i] = special_rand();
        }

        dnn->activation_functions = (activation_function_ptr *)realloc(dnn->activation_functions, sizeof(activation_function_ptr) * dnn->hidden_layers);
        dnn->activation_functions[pos] = activation;

        dnn->gradients = (activation_function_ptr *)realloc(dnn->gradients, sizeof(activation_function_ptr) * dnn->hidden_layers);
        dnn->gradients[pos] = grad_activation;
    }
}

void destroyDNN(DeepNeuralNetwork *dnn)
{
    for (int i = 0; i < dnn->hidden_layers; i++)
    {
        free(dnn->layers_biases[i].data);
        free(dnn->layers_weights[i].data);
    }

    free(dnn->layers_biases);
    free(dnn->layers_weights);
    free(dnn->activation_functions);
    free(dnn->gradients);
    free(dnn);
}

double special_rand()
{
    return ((rand() + 1.0) / (RAND_MAX + 1.0) * 2.0 - 1.0) / 2.0;
}

double relu(double value)
{
    if (value >= 0)
    {
        return value;
    }
    return 0;
}

Matrix *m_relu(Matrix *values)
{
    return apply_function(values, relu);
}

double grad_relu(double value)
{
    if (value > 0)
    {
        return 1;
    }
    else if (value == 0)
    {
        return 0.5;
    }
    else
    {
        return 0;
    }
}

double sigmoid(double x)
{
    float exp_value;
    float return_value;

    exp_value = exp(-x);

    return_value = 1 / (1 + exp_value);

    return return_value;
}

Matrix *m_sigmoid(Matrix *values)
{
    return apply_function(values, sigmoid);
}

double grad_sigmoid(double value)
{
    return sigmoid(value) * (1 - sigmoid(value));
}

Matrix *mse(Matrix *actual, Matrix *expected)
{
    double sum = 0;

    for (uint i = 1; i < actual->rows; i++)
    {
        double diff = get_entry(actual, i, 1) - get_entry(expected, i, 1);
        sum += diff * diff;
    }

    sum = sum / actual->rows;
    sum = sum / 2.0;

    Matrix *result = allocate_matrix(1, 1);
    set_entry(result, 1, 1, sum);

    return result;
}

Matrix *grad_mse(Matrix *actual, Matrix *expected)
{
    Matrix *diff = subtract_matrices(actual, expected);
    Matrix *result = multiply_scalar(diff, 1.0 / actual->rows);
    deallocate_matrix(diff);
    return result;
}

Matrix *grad_m_sigmoid(Matrix *values)
{
    Matrix *grad = apply_function(values, grad_sigmoid);
    Matrix *result = diagonalize(grad);
    deallocate_matrix(grad);
    return result;
}

Matrix *grad_m_relu(Matrix *values)
{
    Matrix *grad = apply_function(values, grad_relu);
    Matrix *result = diagonalize(grad);
    deallocate_matrix(grad);
    return result;
}

Matrix *softmax(Matrix *values)
{
    Matrix *exponent = apply_function(values, exp);
    double sum = 0;
    for (uint i = 1; i <= exponent->rows; i++)
    {
        sum += get_entry(exponent, i, 1);
    }
    Matrix *result = multiply_scalar(exponent, 1.0 / sum);
    deallocate_matrix(exponent);
    return result;
}

Matrix *grad_softmax(Matrix *values)
{
    Matrix *m_softmax = softmax(values);
    Matrix *result = allocate_matrix(values->rows, values->rows);

    for (uint i = 1; i <= values->rows; i++)
    {
        for (uint j = 1; j <= values->rows; j++)
        {
            if (i == j)
            {
                double s_i = get_entry(m_softmax, i, 1);
                set_entry(result, i, i, (1 - s_i) * s_i);
            }
            else
            {
                set_entry(result, i, j, -get_entry(m_softmax, i, 1) * get_entry(m_softmax, j, 1));
            }
        }
    }

    free(m_softmax);
    return result;
}

Matrix *cross_entropy_loss(Matrix *actual, Matrix *expected)
{
    Matrix *result = allocate_matrix(1, 1);

    double sum = 0;
    for (uint i = 1; i <= actual->rows; i++)
    {
        sum += get_entry(expected, i, 1) * log(get_entry(actual, i, 1));
    }

    set_entry(result, 1, 1, -sum);
}

Matrix *grad_cross_entropy(Matrix *actual, Matrix *expected)
{
    Matrix *result = allocate_matrix(actual->rows, 1);

    for (uint i = 1; i <= actual->rows; i++)
    {
        set_entry(result, i, 1, -get_entry(expected, i, 1) / get_entry(actual, i, 1));
    }

    return result;
}