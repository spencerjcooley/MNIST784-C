#include <stdlib.h>
#include <math.h>

#include "matrix.h"
#include "train.h"

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