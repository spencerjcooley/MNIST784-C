#include <stdlib.h>
#include <math.h>
#include <assert.h>

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
float cce_loss(const Matrix *logits, const int *label) {\
    float total_loss = 0.0f;
    
    for (size_t batch = 0; batch < logits->rows; batch++) {
        assert(label[batch] >= 0 && label[batch] < logits->cols);

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
        total_loss += log_sum_exp - logits->data[row_offset + label[batch]];
    }

    return total_loss / (float)logits->rows;
}