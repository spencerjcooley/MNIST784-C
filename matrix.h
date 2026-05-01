#ifndef DATASET_H
#define DATASET_H

typedef struct matrix {
    float *data;
    int rows;
    int cols;
} Matrix;

Matrix create_matrix(int rows, int cols);
void init_matrix(Matrix *matrix);
void free_matrix(Matrix *matrix);

void matmul(const Matrix *A, const Matrix *B, Matrix *out);

#endif