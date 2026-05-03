#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "train.h"
#include "dataset.h"
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
    const int *labels,
    const float lr
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

    sgd_update(&network->W3, &dW3, lr);
    sgd_update(&network->B3, &dB3, lr);
    sgd_update(&network->W2, &dW2, lr);
    sgd_update(&network->B2, &dB2, lr);
    sgd_update(&network->W1, &dW1, lr);
    sgd_update(&network->B1, &dB1, lr);

    free_matrix(&dZ3);
    free_matrix(&dW3);
    free_matrix(&dB3);
    free_matrix(&dA2);
    free_matrix(&dZ2);
    free_matrix(&dW2);
    free_matrix(&dB2);
    free_matrix(&dA1);
    free_matrix(&dZ1);
    free_matrix(&dW1);
    free_matrix(&dB1);
}

void sgd_update(Matrix *M, Matrix *dM, float lr) {
    size_t size = M->rows * M->cols;
    for (size_t i = 0; i < size; i++) {
        M->data[i] -= lr * dM->data[i];
    }
}

void train(
    Network *network,
    Dataset *train,
    size_t batch_size,
    size_t epochs,
    float lr
) {
    size_t n_batches = train->n / batch_size;

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (size_t batch = 0; batch < n_batches; batch++) {
            size_t offset = batch * batch_size;

            Matrix X = wrap_matrix(
                &train->x[offset * IMAGE_SIZE],
                batch_size,
                IMAGE_SIZE
            );
            int *Y = &train->y[offset];

            forward_propagation(&X, network);
            epoch_loss += cce_loss(&network->Z3, Y);
            backward_propagation(network, &X, Y, lr);
        }

        epoch_loss /= n_batches;
        printf("Epoch %03lu | Loss: %.5f\n", epoch, epoch_loss);
    }
}

void test(Network *network, Dataset *test, size_t batch_size) {
    size_t n_batches = test->n / batch_size;

    float total_loss = 0.0f;
    size_t correct = 0;
    size_t total = 0;

    for (size_t batch = 0; batch < n_batches; batch++) {
        size_t offset = batch * batch_size;

        Matrix X = wrap_matrix(
            &test->x[offset * IMAGE_SIZE],
            batch_size,
            IMAGE_SIZE
        );
        int *Y = &test->y[offset];

        forward_propagation(&X, network);
        total_loss += cce_loss(&network->Z3, Y);

        // Accuracy
        for (size_t sample = 0; sample < batch_size; sample++) {
            size_t row_offset = sample * network->A3.cols;
            size_t pred = 0;
            float best = network->A3.data[row_offset];
            for (size_t neuron = 1; neuron < network->A3.cols; neuron++) {
                float v = network->A3.data[row_offset + neuron];
                if (v > best) {
                    best = v;
                    pred = neuron;
                }
            }
            if ((int)pred == Y[sample]) correct++;
        }

        total += batch_size;
    }

    total_loss /= n_batches;
    float accuracy = (float)correct / (float)total * 100.0f;

    printf("TEST | Loss: %.5f | Accuracy: %.2f%%\n", total_loss, accuracy);
}