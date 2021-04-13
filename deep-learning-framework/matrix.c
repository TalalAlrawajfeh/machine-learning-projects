#include "matrix.h"


inline Matrix *allocate_matrix(uint rows,
                               uint columns)
{
    Matrix *result = (Matrix *)malloc(sizeof(Matrix));

    if (result == NULL)
    {
        return NULL;
    }

    result->rows = rows;
    result->columns = columns;
    result->data = (double *)malloc(sizeof(double) * rows * columns);

    return result;
}

Matrix *init_matrix(uint rows,
                    uint columns,
                    double default_value)
{
    Matrix *result = allocate_matrix(rows, columns);

    for (uint i = 0; i < rows * columns; i++)
    {
        result->data[i] = default_value;
    }

    return result;
}

inline void deallocate_matrix(Matrix *matrix)
{
    free(matrix->data);
    free(matrix);
}

inline double get_entry(Matrix *matrix,
                        uint row,
                        uint column)
{
    if (row > matrix->rows || column > matrix->columns)
    {
        return 0;
    }

    return matrix->data[matrix->columns * (row - 1) + column - 1];
}

inline void set_entry(Matrix *matrix,
                      uint row,
                      uint column,
                      double value)
{
    if (row > matrix->rows)
    {
        return;
    }

    matrix->data[matrix->columns * (row - 1) + column - 1] = value;
}

Matrix *matrix_product(Matrix *matrix1,
                       Matrix *matrix2)
{
    if (matrix1->columns != matrix2->rows)
    {
        return NULL;
    }

    uint rows = matrix1->rows;
    uint columns = matrix2->columns;

    Matrix *result = allocate_matrix(rows, columns);

    if (result == NULL)
    {
        return NULL;
    }

    for (uint row_1 = 1; row_1 <= rows; row_1++)
    {
        for (uint column_2 = 1; column_2 <= columns; column_2++)
        {
            double sum = 0;

            for (uint i = 1; i <= matrix1->columns; i++)
            {
                sum += get_entry(matrix1, row_1, i) * get_entry(matrix2, i, column_2);
            }

            set_entry(result, row_1, column_2, sum);
        }
    }

    return result;
}

Matrix *hadamard_product(Matrix *matrix1,
                         Matrix *matrix2)
{
    if (matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns)
    {
        return NULL;
    }

    Matrix *result = allocate_matrix(matrix1->rows, matrix1->columns);

    if (result == NULL)
    {
        return NULL;
    }

    for (uint row = 1; row <= matrix1->rows; row++)
    {
        for (uint column = 1; column <= matrix1->columns; column++)
        {
            set_entry(result, row, column,
                      get_entry(matrix1, row, column) * get_entry(matrix2, row, column));
        }
    }

    return result;
}

Matrix *add_matrices(Matrix *matrix1,
                     Matrix *matrix2)
{
    if (matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns)
    {
        return NULL;
    }

    Matrix *result = allocate_matrix(matrix1->rows, matrix1->columns);

    if (result == NULL)
    {
        return NULL;
    }

    for (uint row = 1; row <= matrix1->rows; row++)
    {
        for (uint column = 1; column <= matrix1->columns; column++)
        {
            set_entry(result, row, column,
                      get_entry(matrix1, row, column) + get_entry(matrix2, row, column));
        }
    }

    return result;
}

Matrix *subtract_matrices(Matrix *matrix1,
                          Matrix *matrix2)
{
    if (matrix1->rows != matrix2->rows || matrix1->columns != matrix2->columns)
    {
        return NULL;
    }

    Matrix *result = allocate_matrix(matrix1->rows, matrix1->columns);

    if (result == NULL)
    {
        return NULL;
    }

    for (uint row = 1; row <= matrix1->rows; row++)
    {
        for (uint column = 1; column <= matrix1->columns; column++)
        {
            set_entry(result, row, column,
                      get_entry(matrix1, row, column) - get_entry(matrix2, row, column));
        }
    }

    return result;
}

Matrix *multiply_scalar(Matrix *matrix,
                        double scalar)
{
    Matrix *result = allocate_matrix(matrix->rows, matrix->columns);

    if (result == NULL)
    {
        return NULL;
    }

    for (uint row = 1; row <= matrix->rows; row++)
    {
        for (uint column = 1; column <= matrix->columns; column++)
        {
            set_entry(result, row, column,
                      get_entry(matrix, row, column) * scalar);
        }
    }

    return result;
}

Matrix *transpose(Matrix *matrix)
{
    Matrix *result = allocate_matrix(matrix->columns, matrix->rows);

    if (result == NULL)
    {
        return NULL;
    }

    for (uint row = 1; row <= matrix->rows; row++)
    {
        for (uint column = 1; column <= matrix->columns; column++)
        {
            set_entry(result, column, row,
                      get_entry(matrix, row, column));
        }
    }

    return result;
}

Matrix *apply_function(Matrix *matrix,
                       double (*func)(double))
{
    Matrix *result = allocate_matrix(matrix->rows, matrix->columns);

    for (uint row = 1; row <= matrix->rows; row++)
    {
        for (uint column = 1; column <= matrix->columns; column++)
        {
            set_entry(result, row, column,
                      func(get_entry(matrix, row, column)));
        }
    }

    return result;
}

Matrix *diagonalize(Matrix *vector)
{
    if (vector->columns > 1)
    {
        return NULL;
    }

    Matrix *result = allocate_matrix(vector->rows, vector->rows);

    for (uint i = 1; i <= vector->rows; i++)
    {
        for (uint j = 1; j <= vector->rows; j++)
        {
            set_entry(result, i, j, 0.0);
        }
    }

    for (uint i = 1; i <= vector->rows; i++)
    {
        set_entry(result, i, i,
                  get_entry(vector, i, 1));
    }

    return result;
}