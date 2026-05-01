#include <stdlib.h>

#include "network.h"

// Applies affine transformation to input matrix X.
void dense_forward(
    const Matrix *X,
    const Matrix *W,
    const Matrix *B,
    Matrix *Y
) {
    if (B->rows != 1) {
        perror("Bias must be a vector.\n");
        return;
    }

    if (X->cols != W->rows || B->cols != W->cols) {
        perror("Incompatible matrix dimensions.\n");
        return;
    }

    Y->rows = X->rows;
    Y->cols = W->cols;

    // Inner dimension.
    const size_t inner = X->cols;
    for (size_t X_row = 0; X_row < X->rows; X_row++) {
        // Initialise output row with biases.
        size_t Y_row = X_row * W->cols;
        for (size_t W_col = 0; W_col < W->cols; W_col++) {
            Y->data[Y_row + W_col] = B->data[W_col];
        }

        // Iterate over inner dimension.
        for (size_t i = 0; i < inner; i++) {
            float X_val = X->data[X_row * X->cols + i];
            size_t W_row = i * W->cols;
            for (size_t W_col = 0; W_col < W->cols; W_col++) {
                Y->data[Y_row + W_col] += X_val * W->data[W_row + W_col];
            }
        }
    }
}