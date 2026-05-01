#ifndef DATASET_H
#define DATASET_H

typedef struct matrix {
    float *data;
    int rows;
    int cols;
} Matrix;

void matmul(const Matrix *A, const Matrix *B, Matrix *out);

#endif