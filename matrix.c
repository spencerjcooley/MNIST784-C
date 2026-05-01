#include <stdlib.h>
#include <math.h>

#include "matrix.h"

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

// He initialisation for a matrix.
void init_matrix(Matrix *matrix, size_t fan_in) {
    float limit = sqrtf(6.0f / fan_in);

    for (size_t i = 0; i < matrix->rows * matrix->cols; i++) {
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