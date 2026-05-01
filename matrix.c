#include <stdlib.h>
#include <math.h>

#include "matrix.h"

// Allocates memory and sets variables for a new Matrix struct.
Matrix create_matrix(int rows, int cols) {
    Matrix output;
    output.rows = rows;
    output.cols = cols;
    output.data = malloc(rows * cols * sizeof(float));

    if (!output.data) {
        perror("Error allocating memory.\n");
        output.rows = 0;
        output.cols = 0;
    }

    return output;
}

// He initialisation for a matrix.
void init_matrix(Matrix *matrix) {
    float limit = sqrtf(6.0f / matrix->rows);

    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        float uniform = (float)rand() / (float)RAND_MAX;
        matrix->data[i] = (2.0f * uniform - 1.0f) * limit;
    }
}

// Free malloc'd memory for a Matrix struct.
void free_matrix(Matrix *matrix) {
    free(matrix->data);
    matrix->data = NULL;
    matrix->rows = 0;
    matrix->cols = 0;
}

// Multiplies matrices A and B together, result in out.
void matmul(const Matrix *A, const Matrix *B, Matrix *out) {
    if (A->cols != B->rows) {
        perror("Incompatible matrix dimensions.\n");
        return;
    }

    out->rows = A->rows;
    out->cols = B->cols;

    // Inner dimension.
    const int inner = A->cols;
    for (int A_row = 0; A_row < A->rows; A_row++) {
        // Initialise output row.
        int out_row = A_row * B->cols;
        for (int B_col = 0; B_col < B->cols; B_col++) {
            out->data[out_row + B_col] = 0.0f;
        }

        // Iterate over inner dimension.
        for (int i = 0; i < inner; i++) {
            float A_val = A->data[A_row * A->cols + i];
            int B_row = i * B->cols;
            for (int B_col = 0; B_col < B->cols; B_col++) {
                out->data[out_row + B_col] += A_val * B->data[B_row + B_col];
            }
        }
    }
}