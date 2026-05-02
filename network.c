#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "network.h"
#include "matrix.h"



Network create_network(size_t batch_size, size_t input_dim) {
    Network network = {
        .W1 = create_matrix(input_dim, 128),
        .B1 = create_matrix(1, 128),
        .Z1 = create_matrix(batch_size, 128),
        .A1 = create_matrix(batch_size, 128),

        .W2 = create_matrix(128, 64),
        .B2 = create_matrix(1, 64),
        .Z2 = create_matrix(batch_size, 64),
        .A2 = create_matrix(batch_size, 64),

        .W3 = create_matrix(64, 10),
        .B3 = create_matrix(1, 10),
        .Z3 = create_matrix(batch_size, 10),
        .A3 = create_matrix(batch_size, 10),
    };

    return network;
}

// Initialise network with random W/B matrices.
void init_network(Network *network) {
    // Initialise weight matrices.
    init_weights(&network->W1, network->W1.rows);
    init_weights(&network->W2, network->W2.rows);
    init_weights(&network->W3, network->W3.rows);

    // Initialise bias vectors.
    init_biases(&network->B1, 0.0f);
    init_biases(&network->B2, 0.0f);
    init_biases(&network->B3, 0.0f);
}

// He uniform initialisation.
void init_weights(Matrix *W, size_t fan_in) {
    float limit = sqrtf(6.0f / fan_in);

    for (size_t i = 0; i < W->rows * W->cols; i++) {
        float uniform = (float)rand() / (float)RAND_MAX;
        W->data[i] = (2.0f * uniform - 1.0f) * limit;
    }
}

// Basic bias initialisation for 0/small val.
void init_biases(Matrix *B, float val) {
    assert(B->rows == 1);
    for (size_t i = 0; i < B->cols; i++) {
        B->data[i] = val;
    }
}

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
    for (size_t i = 0; i < Z->rows * Z->cols; i++) {
        A->data[i] = Z->data[i] > 0 ? Z->data[i] : 0;
    }
}

// Safe Softmax activation function (for output layer).
void softmax(const Matrix *Z, Matrix *A) {
    assert (Z->rows == A->rows && Z->cols == A->cols);

    size_t rows = Z->rows;
    size_t cols = Z->cols;

    for (size_t batch = 0; batch < rows; batch++) {
        size_t row_offset = batch * cols;

        // Calculate max.
        float max = Z->data[row_offset];
        for (size_t i = 1; i < cols; i++) {
            float val = Z->data[row_offset + i];
            max = val > max ? val : max;
        }

        // Calculate all exponentials & get sum.
        float sum = 0.0f;
        for (size_t i = 0; i < cols; i++) {
            float e = expf(Z->data[row_offset + i] - max);
            A->data[row_offset + i] = e;
            sum += e;
        }

        // Divide by the exponentiated sum.
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < cols; i++) {
            A->data[row_offset + i] *= inv_sum;
        }
    }
}

void forward_propagation(const Matrix *input, Network *network) {
    dense_forward(input, &network->W1, &network->B1, &network->Z1);
    relu(&network->Z1, &network->A1);

    dense_forward(&network->A1, &network->W2, &network->B2, &network->Z2);
    relu(&network->Z2, &network->A2);

    dense_forward(&network->A2, &network->W3, &network->B3, &network->Z3);
    softmax(&network->Z3, &network->A3);
}

void free_network(Network *network) {
    free_matrix(&network->W1);
    free_matrix(&network->B1);
    free_matrix(&network->Z1);
    free_matrix(&network->A1);

    free_matrix(&network->W2);
    free_matrix(&network->B2);
    free_matrix(&network->Z2);
    free_matrix(&network->A2);

    free_matrix(&network->W3);
    free_matrix(&network->B3);
    free_matrix(&network->Z3);
    free_matrix(&network->A3);
}