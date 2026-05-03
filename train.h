#ifndef TRAIN_H
#define TRAIN_H

#include <stdlib.h>

#include "matrix.h"
#include "network.h"
#include "dataset.h"


void forward_propagation(const Matrix *input, Network *network);

float cce_loss(const Matrix *logits, const int *labels);
void backward_propagation(
    Network *network,
    const Matrix *input,
    const int *labels,
    const float lr
);
void sgd_update(Matrix *M, Matrix *dM, float lr);

void train(
    Network *network,
    Dataset *train,
    size_t batch_size,
    size_t epochs,
    float lr
);

#endif