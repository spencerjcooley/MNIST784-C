#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "train.h"
#include "matrix.h"
#include "network.h"

void forward_propagation(const Matrix *input, Network *network) {
    dense_forward(input, &network->W1, &network->B1, &network->Z1);
    relu(&network->Z1, &network->A1);

    dense_forward(&network->A1, &network->W2, &network->B2, &network->Z2);
    relu(&network->Z2, &network->A2);

    dense_forward(&network->A2, &network->W3, &network->B3, &network->Z3);
    softmax(&network->Z3, &network->A3);
}

// Categorical Cross Entropy Loss (for one hot encoded output layer).
float cce_loss(const Matrix *logits, const int *labels) {\
    float total_loss = 0.0f;
    
    for (size_t batch = 0; batch < logits->rows; batch++) {
        assert(labels[batch] >= 0 && labels[batch] < logits->cols);

        size_t row_offset = batch * logits->cols;
        float max = logits->data[row_offset];
        for (size_t neuron = 1; neuron < logits->cols; neuron++) {
            float val = logits->data[row_offset + neuron];
            max = val > max ? val : max;
        }

        float sum = 0.0f;
        for (size_t neuron = 0; neuron < logits->cols; neuron++) {
            sum += expf(logits->data[row_offset + neuron] - max);
        }

        float log_sum_exp = max + logf(sum);
        total_loss += log_sum_exp - logits->data[row_offset + labels[batch]];
    }

    return total_loss / (float)logits->rows;
}

void backward_propagation(
    Network *network,
    const Matrix *input,
    const int *labels
) {
    size_t batch_size = network->Z3.rows;

    // Softmax backward.
    Matrix dZ3 = create_matrix(network->Z3.rows, network->Z3.cols);
    float inv_batch_size = 1.0f / batch_size;
    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t neuron = 0; neuron < network->A3.cols; neuron++) {
            size_t index = batch * network->A3.cols + neuron;
            dZ3.data[index] = network->A3.data[index] * inv_batch_size;
        }
        dZ3.data[batch * network->A3.cols + labels[batch]] -= inv_batch_size;
    }

    Matrix dW3 = create_matrix(network->W3.rows, network->W3.cols);
    Matrix dB3 = create_matrix(network->B3.rows, network->B3.cols);
    Matrix dA2 = create_matrix(network->A2.rows, network->A2.cols);
    Matrix dZ2 = create_matrix(network->Z2.rows, network->Z2.cols);
    dense_backward(&dZ3, &network->A2, &network->W3, &dW3, &dB3, &dA2);
    relu_backward(&dA2, &network->Z2, &dZ2);

    Matrix dW2 = create_matrix(network->W2.rows, network->W2.cols);
    Matrix dB2 = create_matrix(network->B2.rows, network->B2.cols);
    Matrix dA1 = create_matrix(network->A1.rows, network->A1.cols);
    Matrix dZ1 = create_matrix(network->Z1.rows, network->Z1.cols);
    dense_backward(&dZ2, &network->A1, &network->W2, &dW2, &dB2, &dA1);
    relu_backward(&dA1, &network->Z1, &dZ1);

    Matrix dW1 = create_matrix(network->W1.rows, network->W1.cols);
    Matrix dB1 = create_matrix(network->B1.rows, network->B1.cols);
    dense_backward(&dZ1, input, &network->W1, &dW1, &dB1, NULL);
}