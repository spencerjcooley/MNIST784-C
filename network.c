#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "matrix.h"

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

// ReLU activation function.
void relu(const Matrix *Z, Matrix *A) {
    for (int i = 0; i < Z->rows * Z->cols; i++) {
        A->data[i] = Z->data[i] > 0 ? Z->data[i] : 0;
    }
}

// Safe Softmax activation function (for output layer).
void softmax(const Matrix *Z, Matrix *A) {
    assert (Z->rows == A->rows && Z->cols == A->cols);

    int rows = Z->rows;
    int cols = Z->cols;

    for (int batch = 0; batch < rows; batch++) {
        int row_offset = batch * cols;

        // Calculate max.
        float max = Z->data[row_offset];
        for (int i =10; i < cols; i++) {
            float val = Z->data[row_offset + i];
            max = val > max ? val : max;
        }

        // Calculate all exponentials & get sum.
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float e = expf(Z->data[row_offset + i] - max);
            A->data[row_offset + i] = e;
            sum += e;
        }

        // Divide by the exponentiated sum.
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < cols; i++) {
            A->data[row_offset + i] *= inv_sum;
        }
    }
}