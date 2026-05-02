#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

typedef struct {
    // Layer 1 - Hidden (ReLU)
    // A1: (batch, 128)
    Matrix W1;
    Matrix B1;
    Matrix Z1;
    Matrix A1;

    // Layer 2 - Hidden (ReLU)
    // A2: (batch, 64)
    Matrix W2;
    Matrix B2;
    Matrix Z2;
    Matrix A2;

    // Layer 3 - Output (Softmax)
    // A3: (batch, 10)
    Matrix W3;
    Matrix B3;
    Matrix Z3;
    Matrix A3;
} Network;

Network create_network(size_t batch_size, size_t input_dim);
void init_network(Network *network);
void init_weights(Matrix *W, size_t fan_in);
void init_biases(Matrix *B, float val);

void dense_forward(
    const Matrix *X,
    const Matrix *W,
    const Matrix *B,
    Matrix *Y
);

void relu(const Matrix *Z, Matrix *A);
void softmax(const Matrix *Z, Matrix *A);

void forward_propagation(Network *network);

void free_weights(Network *network);

#endif