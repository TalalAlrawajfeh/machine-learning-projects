#include "dnn.h"
#include "mnist.h"

void represent_matrix(Matrix *matrix,
                      uint precision)
{
    char format[8];
    snprintf(format, 7, "%%.%df\t", precision);

    for (uint row = 1; row <= matrix->rows; row++)
    {
        for (uint column = 1; column <= matrix->columns; column++)
        {
            printf(format, get_entry(matrix, row, column));
        }
        printf("\n");
    }
}

int arg_max(Matrix *vector)
{
    double max = -DBL_MAX;
    int max_i = -1;

    for (int i = 0; i < vector->rows; i++)
    {
        double entry = get_entry(vector, i + 1, 1);
        if (entry > max)
        {
            max = entry;
            max_i = i;
        }
    }

    return max_i;
}

void test()
{
    load_mnist();

    DeepNeuralNetwork *dnn = createDNN(784);
    add_layer(dnn, 784, m_relu, grad_m_relu);
    add_layer(dnn, 10, softmax, grad_softmax);

    Matrix *inputs = (Matrix *)malloc(sizeof(Matrix) * NUM_TRAIN);
    Matrix *outputs = (Matrix *)malloc(sizeof(Matrix) * NUM_TRAIN);

    for (uint i = 0; i < NUM_TRAIN; i++)
    {
        inputs[i].columns = 1;
        inputs[i].rows = 784;
        inputs[i].data = &train_image[i];

        outputs[i].columns = 1;
        outputs[i].rows = 10;
        outputs[i].data = (double *)malloc(sizeof(double) * 10);
        for (uint j = 0; j < 10; j++)
        {
            outputs[i].data[j] = 0;
        }
        outputs[i].data[train_label[i]] = 1.0;
    }

    for (uint epoch = 1; epoch <= 2; epoch++)
    {
        for (uint batch = 0; batch < 1875; batch++)
        {
            printf("\repoch: %d, batch: %d                ", epoch, batch + 1);
            train_minibatch(dnn,
                            inputs + batch * 32,
                            outputs + batch * 32,
                            32,
                            cross_entropy_loss,
                            grad_cross_entropy,
                            0.01);
        }
    }

    printf("\n");
    
    Matrix input;
    input.rows = 784;
    input.columns = 1;

    int correct = 0;
    for (int i = 0; i < NUM_TEST; i++)
    {
        input.data = &train_image[i];
        Matrix *result = feed_forward(dnn, &input);
        int predicted = arg_max(result);
        int actual = train_label[i];
        if (predicted == actual)
        {
            correct++;
        }
        deallocate_matrix(result);
    }

    double accuracy = (correct * 1.0) / (NUM_TEST * 1.0);
    printf("Test Accuracy: %f\n", accuracy);

    destroyDNN(dnn);
}

int main(int argc, char *argv)
{
    test();
    return 0;
}