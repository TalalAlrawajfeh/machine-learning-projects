#ifndef MATRIX_H
#define MATRIX_H

#include <malloc.h>
#include <math.h>
#include <float.h>

typedef unsigned int uint;

typedef struct _Matrix
{
    double *data;
    uint rows;
    uint columns;
} Matrix;

extern inline Matrix *allocate_matrix(uint rows,
                                      uint columns);

extern inline void deallocate_matrix(Matrix *matrix);

extern inline double get_entry(Matrix *matrix,
                               uint row,
                               uint column);

extern inline void set_entry(Matrix *matrix,
                             uint row,
                             uint column,
                             double value);

Matrix *init_matrix(uint rows,
                    uint columns,
                    double default_value);

inline void deallocate_matrix(Matrix *matrix);

Matrix *matrix_product(Matrix *matrix1,
                       Matrix *matrix2);

Matrix *hadamard_product(Matrix *matrix1,
                         Matrix *matrix2);

Matrix *add_matrices(Matrix *matrix1,
                     Matrix *matrix2);

Matrix *subtract_matrices(Matrix *matrix1,
                          Matrix *matrix2);

Matrix *multiply_scalar(Matrix *matrix,
                        double scalar);

Matrix *transpose(Matrix *matrix);

Matrix *apply_function(Matrix *matrix,
                       double (*func)(double));

Matrix *diagonalize(Matrix *vector);

#endif /* MATRIX_H */