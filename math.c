#include <stdlib.h>
#include "math.h"

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