#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix {
    float *data;
    size_t rows;
    size_t cols;
} Matrix;

Matrix wrap_matrix(float *data, size_t rows, size_t cols);
Matrix create_matrix(size_t rows, size_t cols);
void free_matrix(Matrix *matrix);

#endif