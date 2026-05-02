#ifndef TRAIN_H
#define TRAIN_H

#include <stdlib.h>

#include "matrix.h"
#include "network.h"

typedef struct {
    float learning_rate;
    size_t epochs;
    size_t batch_size;
} HyperParams;

float cce_loss(const Matrix *logits, const int *labels);

#endif