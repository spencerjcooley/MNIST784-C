#include <stdlib.h>
#include <math.h>

#include "matrix.h"

// Wraps a row-major array in Matrix struct (for batching).
// Avoids unnecessary mallocs.
Matrix wrap_matrix(float *data, size_t rows, size_t cols) {
    Matrix output = {
        .data = data,
        .rows = rows,
        .cols = cols
    };

    return output;
}

// Allocates memory and sets variables for a new Matrix struct.
Matrix create_matrix(size_t rows, size_t cols) {
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

// Free malloc'd memory for a Matrix struct (ONLY ON CREATED, NOT WRAPPED)
void free_matrix(Matrix *matrix) {
    free(matrix->data);
    matrix->data = NULL;
    matrix->rows = 0;
    matrix->cols = 0;
}