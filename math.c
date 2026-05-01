#include <stdlib.h>
#include "math.h"

void matmul(const Matrix *A, const Matrix *B, Matrix *out) {
    if (A->cols != B->rows) {
        perror("Matrices have incompatible dimensions\n");
        return;
    }

    out->rows = A->rows;
    out->cols = B->cols;

    // Inner dimension.
    const int inner = A->cols;
    for (int A_row = 0; A_row < A->rows; A_row++) {
        for (int B_col = 0; B_col < B->cols; B_col++) {

            float sum = 0.0f;
            // Iterates over both matrices A and B in their row/col.
            for (int i = 0; i < inner; i++) {
                sum += A->data[A_row * A->cols + i] * B->data[i * B->cols + B_col];
            }

            out->data[A_row * inner + B_col];

        }
    }
}